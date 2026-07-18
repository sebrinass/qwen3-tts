#pragma once
// codec-chunked-decode.h: bounded VRAM codec decode with rolling left
// context. Strict equivalent of the upstream Qwen3-TTS 12 Hz tokenizer
// chunked_decode entry
// (qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py
// line 886).
//
// The codec decoder runs a causal Conv1d, a sliding window causal
// transformer, an upsample stage and a DAC decoder. Decoding a chunk
// of frames in isolation introduces edge artefacts at the chunk
// boundary because the causal conv kernel and the transformer attention
// have no left context to draw from. Prepending left_ctx_frames
// previously decoded frames and stripping the resulting samples after
// the decode restores continuity.
//
// Two entry points:
//
//   codec_chunked_decode : one shot decode of a full codes buffer.
//     Bit perfect equivalent of pipeline_codec_decode when the audio
//     fits in a single chunk_frames sized window. Bounds VRAM beyond
//     that, mirrors the upstream chunked_decode loop frame for frame.
//
//   codec_stream_decoder : stateful frame by frame AR streaming.
//     The pipeline pushes one frame at a time as the talker produces
//     them ; push_frame decodes and emits a fresh chunk_frames sized
//     audio block through the on_chunk callback as soon as enough new
//     frames have accumulated. flush drains the tail at EOS.

#include "pipeline-codec.h"
#include "qwen.h"

#include <cstdint>
#include <cstring>
#include <vector>

// One shot chunked decode. codes is K major [K, T] row major (T fastest).
// Returns audio of length T * TOKENIZER_HOP_LENGTH on success, empty on
// failure. chunk_frames clamps to 1, left_ctx_frames clamps to 0.
static inline std::vector<float> codec_chunked_decode(PipelineCodec * pc,
                                                      const int32_t * codes,
                                                      int             K,
                                                      int             T,
                                                      int             chunk_frames,
                                                      int             left_ctx_frames) {
    std::vector<float> out;
    if (T <= 0) {
        return out;
    }
    if (chunk_frames < 1) {
        chunk_frames = 1;
    }
    if (left_ctx_frames < 0) {
        left_ctx_frames = 0;
    }
    out.reserve((size_t) T * (size_t) TOKENIZER_HOP_LENGTH);

    int start = 0;
    while (start < T) {
        int end = start + chunk_frames;
        if (end > T) {
            end = T;
        }
        // Upstream rule : context_size collapses to start when
        // left_ctx_frames would underflow before frame 0.
        int ctx         = (start - left_ctx_frames > 0) ? left_ctx_frames : start;
        int slice_start = start - ctx;
        int slice_T     = end - slice_start;

        std::vector<int32_t> slice((size_t) K * (size_t) slice_T);
        for (int k = 0; k < K; k++) {
            std::memcpy(slice.data() + (size_t) k * (size_t) slice_T,
                        codes + (size_t) k * (size_t) T + (size_t) slice_start, (size_t) slice_T * sizeof(int32_t));
        }
        std::vector<float> wav = pipeline_codec_decode(pc, slice.data(), K, slice_T);
        if (wav.empty()) {
            return std::vector<float>();
        }
        const size_t drop = (size_t) ctx * (size_t) TOKENIZER_HOP_LENGTH;
        if (wav.size() > drop) {
            out.insert(out.end(), wav.begin() + drop, wav.end());
        }
        start = end;
    }
    return out;
}

// Stateful streaming decoder over pipeline_codec_decode_stream with an
// adaptive chunk ramp: the first pushed frame decodes immediately for
// the lowest first byte latency, then chunks grow 2 -> 4 -> 8 frames so
// the steady state interleaves the codec with the talker 8x less often
// and the batch amortizes the kernel count. drain flushes the sub chunk
// tail at EOS with greedy width classes. ICL priming feeds the full
// reference through the same state in max width chunks with the audio
// discarded.
struct codec_stream_decoder {
    int  K;
    // Set true when a flush returned false because the on_chunk
    // callback requested a cancel. Stays false on decode failures so
    // the caller can route to QT_STATUS_CANCELLED vs
    // QT_STATUS_GENERATE_FAILED on a negative return.
    bool cancelled;

    static const int MAX_CHUNK = 1 << (CODEC_STREAM_CLASSES - 1);

    int                  target;     // current ramp chunk width
    int                  pending_n;  // frames accumulated, < target
    std::vector<int32_t> pending;    // [MAX_CHUNK, K] frame major
    std::vector<int32_t> scratch;    // [T, K] chunk upload layout
    std::vector<float>   frame;      // [MAX_CHUNK * hop] audio out

    // Reset the persistent codec state to the zero context and restart
    // the chunk ramp. Returns false when the state allocation fails.
    bool init(PipelineCodec * pc, int K_) {
        K         = K_;
        cancelled = false;
        target    = 1;
        pending_n = 0;
        pending.assign((size_t) MAX_CHUNK * (size_t) K, 0);
        scratch.assign((size_t) MAX_CHUNK * (size_t) K, 0);
        frame.assign((size_t) MAX_CHUNK * (size_t) TOKENIZER_HOP_LENGTH, 0.0f);
        return pipeline_codec_stream_reset(pc);
    }

    // Decode and emit the first T pending frames, then compact the
    // remainder to the front. The scratch reorders frame major pending
    // rows into the [T, K] chunk layout (T contiguous per codebook).
    bool flush_front(PipelineCodec * pc, int T, qt_audio_chunk_cb cb, void * cb_ud) {
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                scratch[(size_t) k * (size_t) T + (size_t) t] = pending[(size_t) t * (size_t) K + (size_t) k];
            }
        }
        if (!pipeline_codec_decode_stream(pc, scratch.data(), T, frame.data())) {
            return false;
        }
        if (cb && !cb(frame.data(), T * TOKENIZER_HOP_LENGTH, cb_ud)) {
            cancelled = true;
            return false;
        }
        pending_n -= T;
        if (pending_n > 0) {
            std::memmove(pending.data(), pending.data() + (size_t) T * (size_t) K,
                         (size_t) pending_n * (size_t) K * sizeof(int32_t));
        }
        return true;
    }

    // Prime the codec state with the full ICL reference: every frame
    // runs through the streaming decode in max width chunks with the
    // audio discarded, so the first generated frame sees the
    // reference's exact causal state. A reference already primed
    // restores its snapshot device to device instead of re-decoding; a
    // fresh one saves its primed state into the LRU. ref_kt is K major
    // [K, ref_T]. Call once, after init and before any push_frame.
    bool seed_reference(PipelineCodec * pc, const int32_t * ref_kt, int ref_T) {
        const uint64_t key = pipeline_codec_ref_key(ref_kt, K, ref_T);
        if (pipeline_codec_stream_restore(pc, key)) {
            return true;
        }
        int t0 = 0;
        while (t0 < ref_T) {
            int T = MAX_CHUNK;
            while (T > ref_T - t0) {
                T >>= 1;
            }
            for (int k = 0; k < K; k++) {
                for (int t = 0; t < T; t++) {
                    scratch[(size_t) k * (size_t) T + (size_t) t] =
                        ref_kt[(size_t) k * (size_t) ref_T + (size_t) (t0 + t)];
                }
            }
            if (!pipeline_codec_decode_stream(pc, scratch.data(), T, NULL)) {
                return false;
            }
            t0 += T;
        }
        return pipeline_codec_stream_snapshot(pc, key);
    }

    // Accumulate one frame (K int32 codes, one per codebook) and decode
    // when the ramp chunk fills. Returns false on decode failure or
    // when cb returns false (cancellation).
    bool push_frame(PipelineCodec * pc, const int32_t * frame_codes, qt_audio_chunk_cb cb, void * cb_ud) {
        std::memcpy(pending.data() + (size_t) pending_n * (size_t) K, frame_codes, (size_t) K * sizeof(int32_t));
        pending_n++;
        if (pending_n < target) {
            return true;
        }
        if (!flush_front(pc, target, cb, cb_ud)) {
            return false;
        }
        if (target < MAX_CHUNK) {
            target <<= 1;
        }
        return true;
    }

    // Flush the sub chunk tail at EOS through greedy width classes.
    bool drain(PipelineCodec * pc, qt_audio_chunk_cb cb, void * cb_ud) {
        while (pending_n > 0) {
            int T = MAX_CHUNK;
            while (T > pending_n) {
                T >>= 1;
            }
            if (!flush_front(pc, T, cb, cb_ud)) {
                return false;
            }
        }
        return true;
    }
};

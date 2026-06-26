// tts-server.cpp: OpenAI-compatible HTTP server backed by the qwentts
// ABI. Loads a talker + codec once, GPU resident, and serves synthesis over
// POST /v1/audio/speech. The shared core lives in src/tts-server.h ; this
// file only wires the qt_* ABI into the generic adapter.
//
// Voice cloning: POST /v1/voices/clone extracts a speaker embedding + RVQ
// codes from an uploaded reference WAV and persists them as .spk / .rvq under
// VOICES_DIR (env, default "./voices"). The speech endpoint resolves voice
// names against builtin speakers first, then falls back to cloned voices.
// DELETE /v1/voices/:voice_id removes a cloned voice. GET /v1/voices lists
// both builtin and cloned voices.

#include "tts-server.h"

#include "qwen.h"
#include "rvq-file.h"
#include "utf8.h"
#include "version.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <direct.h>  // _wmkdir
// <io.h> and <windows.h> are already pulled in by httplib.h / utf8.h
#else
#    include <dirent.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#endif

// 11 bits per RVQ code (codebook size <= 2048), matching qwen-codec / qwen-tts.
static const int RVQ_CODE_BITS = 11;

// ---------------------------------------------------------------------------
// UTF-8 helpers (Windows only; POSIX is UTF-8 by convention)
// ---------------------------------------------------------------------------

#if defined(_WIN32)
static std::string wide_to_utf8(const wchar_t * w) {
    if (!w || !*w) {
        return "";
    }
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (len <= 0) {
        return "";
    }
    std::string s((size_t) len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], len, NULL, NULL);
    return s;
}
#endif

// ---------------------------------------------------------------------------
// Voices directory helpers
// ---------------------------------------------------------------------------

// VOICES_DIR environment variable, default "./voices".
static std::string voices_dir_get() {
    const char * env = std::getenv("VOICES_DIR");
    return (env && *env) ? std::string(env) : std::string("./voices");
}

// Create directory if it doesn't exist. Returns true if the directory exists
// after the call (created or already existed).
static bool voices_dir_ensure(const std::string & dir) {
#if defined(_WIN32)
    std::wstring wdir = utf8_to_wide(dir.c_str());
    if (_wmkdir(wdir.c_str()) == 0) {
        return true;
    }
    return errno == EEXIST;
#else
    if (mkdir(dir.c_str(), 0755) == 0) {
        return true;
    }
    return errno == EEXIST;
#endif
}

// Build a voice file path: {dir}/{voice_id}{ext}
static std::string voice_path(const std::string & dir, const std::string & voice_id, const char * ext) {
    return dir + "/" + voice_id + ext;
}

// Check if a cloned voice exists (.spk file present).
static bool voice_exists(const std::string & dir, const std::string & voice_id) {
    std::string spk = voice_path(dir, voice_id, ".spk");
    FILE *      f   = utf8_fopen(spk.c_str(), "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

// Validate voice_id: non-empty, no path separators, no ".." traversal.
static bool voice_id_valid(const std::string & id) {
    if (id.empty()) {
        return false;
    }
    if (id.find('/') != std::string::npos) {
        return false;
    }
    if (id.find('\\') != std::string::npos) {
        return false;
    }
    if (id.find("..") != std::string::npos) {
        return false;
    }
    return true;
}

// Load .spk file: raw f32 values whose count IS the embedding dimension.
static bool load_spk_file(const std::string & path, std::vector<float> & emb) {
    FILE * f = utf8_fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % (long) sizeof(float)) != 0) {
        fclose(f);
        return false;
    }
    emb.resize((size_t) sz / sizeof(float));
    size_t nr = fread(emb.data(), sizeof(float), emb.size(), f);
    fclose(f);
    return nr == emb.size();
}

// Load .rvq file via rvq-file.h, unpacking K*T codes.
static bool load_rvq_file(const std::string & path, int K, std::vector<int32_t> & codes, int * ref_T) {
    return rvq_read_file(path.c_str(), K, RVQ_CODE_BITS, codes, ref_T);
}

// Load .txt file (optional reference transcript). Trims trailing newlines.
static bool load_txt_file(const std::string & path, std::string & out) {
    FILE * f = utf8_fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        return false;
    }
    out.resize((size_t) sz);
    if (sz > 0 && fread(&out[0], 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f);
        return false;
    }
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

// Save .spk file: raw f32 values.
static bool save_spk_file(const std::string & path, const float * emb, int dim) {
    FILE * f = utf8_fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    size_t nw = fwrite(emb, sizeof(float), (size_t) dim, f);
    fclose(f);
    return nw == (size_t) dim;
}

// Save .txt file.
static bool save_txt_file(const std::string & path, const std::string & text) {
    FILE * f = utf8_fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    bool ok = text.empty() || fwrite(text.data(), 1, text.size(), f) == text.size();
    fclose(f);
    return ok;
}

// Delete a file. Returns true on success (or if the file didn't exist).
static bool file_delete(const std::string & path) {
#if defined(_WIN32)
    std::wstring wpath = utf8_to_wide(path.c_str());
    if (_wremove(wpath.c_str()) == 0) {
        return true;
    }
    return errno == ENOENT;
#else
    if (remove(path.c_str()) == 0) {
        return true;
    }
    return errno == ENOENT;
#endif
}

// List cloned voice IDs by scanning .spk files in the voices directory.
static std::vector<std::string> list_cloned_voices(const std::string & dir) {
    std::vector<std::string> result;
#if defined(_WIN32)
    std::string   pattern = dir + "/*.spk";
    std::wstring  wpattern = utf8_to_wide(pattern.c_str());
    _wfinddata_t  fd;
    intptr_t      handle = _wfindfirst(wpattern.c_str(), &fd);
    if (handle == -1) {
        return result;
    }
    do {
        std::string name = wide_to_utf8(fd.name);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".spk") == 0) {
            result.push_back(name.substr(0, name.size() - 4));
        }
    } while (_wfindnext(handle, &fd) == 0);
    _findclose(handle);
#else
    DIR *          d = opendir(dir.c_str());
    if (!d) {
        return result;
    }
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".spk") == 0) {
            result.push_back(name.substr(0, name.size() - 4));
        }
    }
    closedir(d);
#endif
    std::sort(result.begin(), result.end());
    return result;
}

// Decode WAV bytes from a multipart upload into mono float32 PCM at 24 kHz.
// Returns a malloc'd buffer (caller frees), or NULL on failure.
static float * wav_bytes_to_mono_24k(const char * data, size_t size, int * n_samples) {
    *n_samples = 0;
    int     T  = 0, sr = 0;
    float * planar = audio_io_read_wav_buf((const uint8_t *) data, size, &T, &sr);
    if (!planar || T <= 0) {
        free(planar);
        return nullptr;
    }
    // Resample planar stereo [L:T][R:T] to 24 kHz if needed.
    float * stereo = planar;
    int     T_rs   = T;
    if (sr != 24000) {
        int     T_new     = 0;
        float * resampled = audio_resample(planar, T, sr, 24000, 2, &T_new);
        free(planar);
        if (!resampled || T_new <= 0) {
            free(resampled);
            return nullptr;
        }
        stereo = resampled;
        T_rs   = T_new;
    }
    // Downmix planar stereo to mono = 0.5 * (L + R).
    float * mono = (float *) malloc((size_t) T_rs * sizeof(float));
    if (!mono) {
        free(stereo);
        return nullptr;
    }
    for (int i = 0; i < T_rs; i++) {
        mono[i] = 0.5f * (stereo[i] + stereo[T_rs + i]);
    }
    free(stereo);
    *n_samples = T_rs;
    return mono;
}

// ---------------------------------------------------------------------------
// HTTP handlers for voice management
// ---------------------------------------------------------------------------

// POST /v1/voices/clone : extract voice from uploaded WAV and persist
// .spk / .rvq / .txt under VOICES_DIR.
static void handle_voice_clone(qt_context *              q,
                               const std::string &       voices_dir,
                               const httplib::Request &  req,
                               httplib::Response &       res) {
    if (!req.is_multipart_form_data()) {
        tts_json_error(res, 400, "invalid_request_error", "request must be multipart/form-data");
        return;
    }
    if (!req.form.has_file("file")) {
        tts_json_error(res, 400, "invalid_request_error", "missing 'file' field");
        return;
    }
    std::string voice_id = req.form.has_field("voice_id") ? req.form.get_field("voice_id") : "";
    if (!voice_id_valid(voice_id)) {
        tts_json_error(res, 400, "invalid_request_error",
                       "'voice_id' must be a non-empty string without path separators");
        return;
    }
    // 409 Conflict if voice already exists.
    if (voice_exists(voices_dir, voice_id)) {
        tts_json_error(res, 409, "conflict_error", "voice_id already exists");
        return;
    }

    httplib::FormData file     = req.form.get_file("file");
    std::string       ref_text = req.form.has_field("ref_text") ? req.form.get_field("ref_text") : "";

    // Decode WAV bytes to mono 24 kHz float32.
    int     n_samples = 0;
    float * audio_24k = wav_bytes_to_mono_24k(file.content.data(), file.content.size(), &n_samples);
    if (!audio_24k || n_samples <= 0) {
        free(audio_24k);
        tts_json_error(res, 400, "invalid_request_error", "cannot decode WAV file");
        return;
    }

    // Extract voice reference via the C API (speaker encoder + codec encode).
    struct qt_voice_ref vref = {};
    enum qt_status       rc;
    {
        std::lock_guard<std::mutex> lock(g_synth_mutex);
        rc = qt_extract_voice_ref(q, audio_24k, n_samples, &vref);
    }
    free(audio_24k);
    if (rc != QT_STATUS_OK) {
        tts_json_error(res, tts_status_to_http((int) rc), "server_error", qt_last_error());
        qt_voice_ref_free(&vref);
        return;
    }

    // Persist .spk (raw f32 embedding).
    std::string spk_path = voice_path(voices_dir, voice_id, ".spk");
    if (!save_spk_file(spk_path, vref.ref_spk_emb, vref.ref_spk_dim)) {
        qt_voice_ref_free(&vref);
        tts_json_error(res, 500, "server_error", "failed to save .spk file");
        return;
    }

    // Persist .rvq (packed RVQ codes). ref_codes is [num_codebooks, ref_T] row-major.
    std::string rvq_path = voice_path(voices_dir, voice_id, ".rvq");
    std::vector<int32_t> codes_flat(
        vref.ref_codes,
        vref.ref_codes + (size_t) vref.num_codebooks * (size_t) vref.ref_T);
    if (!rvq_write_file(rvq_path.c_str(), codes_flat, RVQ_CODE_BITS)) {
        file_delete(spk_path);
        qt_voice_ref_free(&vref);
        tts_json_error(res, 500, "server_error", "failed to save .rvq file");
        return;
    }

    // Persist .txt (optional reference transcript for ICL mode B).
    if (!ref_text.empty()) {
        save_txt_file(voice_path(voices_dir, voice_id, ".txt"), ref_text);
    }

    qt_voice_ref_free(&vref);

    // Response: { "voice_id": "...", "status": "created" }
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "voice_id", voice_id.c_str());
    yyjson_mut_obj_add_str(doc, root, "status", "created");
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

// DELETE /v1/voices/:voice_id : remove a cloned voice.
static void handle_voice_delete(const std::string &       voices_dir,
                                const httplib::Request &  req,
                                httplib::Response &       res) {
    std::string voice_id = req.path_params.count("voice_id") ? req.path_params.at("voice_id") : "";
    if (!voice_id_valid(voice_id)) {
        tts_json_error(res, 400, "invalid_request_error", "invalid voice_id");
        return;
    }
    if (!voice_exists(voices_dir, voice_id)) {
        tts_json_error(res, 404, "not_found_error", "voice_id not found");
        return;
    }
    file_delete(voice_path(voices_dir, voice_id, ".spk"));
    file_delete(voice_path(voices_dir, voice_id, ".rvq"));
    file_delete(voice_path(voices_dir, voice_id, ".txt"));

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "voice_id", voice_id.c_str());
    yyjson_mut_obj_add_str(doc, root, "status", "deleted");
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

// GET /v1/voices : list builtin and cloned voices.
static void handle_voices_list(const tts_backend &       be,
                               const std::string &       voices_dir,
                               const httplib::Request &  /*req*/,
                               httplib::Response &       res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    // Builtin speakers from the loaded model.
    yyjson_mut_val * builtin = yyjson_mut_arr(doc);
    for (const std::string & v : be.voices) {
        yyjson_mut_arr_add_str(doc, builtin, v.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "builtin", builtin);

    // Cloned voices scanned from VOICES_DIR.
    yyjson_mut_val *              cloned        = yyjson_mut_arr(doc);
    std::vector<std::string>      cloned_voices = list_cloned_voices(voices_dir);
    for (const std::string & v : cloned_voices) {
        yyjson_mut_arr_add_str(doc, cloned, v.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "cloned", cloned);

    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

// ---------------------------------------------------------------------------
// CLI usage and entry point
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", QWEN_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n\n"
            "Optional:\n"
            "  --host <ip>             Listen address (default: 127.0.0.1)\n"
            "  --port <n>              Listen port (default: 8080)\n"
            "  --lang <name>           Language label (default: auto)\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n\n"
            "Environment:\n"
            "  VOICES_DIR              Directory for cloned voice profiles (default: ./voices)\n",
            prog);
}

// Trim a path down to its file name for the reported model id.
static std::string basename_of(const char * path) {
    std::string s = path;
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);

    const char *  talker_path = NULL;
    const char *  codec_path  = NULL;
    std::string   lang        = "auto";
    server_config cfg;
    bool          use_fa     = true;
    bool          clamp_fp16 = false;

    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (!std::strcmp(arg, "--model") && i + 1 < argc) {
            talker_path = argv[++i];
        } else if (!std::strcmp(arg, "--codec") && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (!std::strcmp(arg, "--host") && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if (!std::strcmp(arg, "--port") && i + 1 < argc) {
            cfg.port = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--lang") && i + 1 < argc) {
            lang = argv[++i];
        } else if (!std::strcmp(arg, "--no-fa")) {
            use_fa = false;
        } else if (!std::strcmp(arg, "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!std::strcmp(arg, "--help") || !std::strcmp(arg, "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!talker_path || !codec_path) {
        print_usage(argv[0]);
        return 0;
    }

    struct qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = talker_path;
    iparams.codec_path  = codec_path;
    iparams.use_fa      = use_fa;
    iparams.clamp_fp16  = clamp_fp16;

    struct qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[Server] FATAL: %s\n", qt_last_error());
        return 1;
    }

    // Ensure the voices directory exists for cloned voice persistence.
    std::string voices_dir = voices_dir_get();
    if (!voices_dir_ensure(voices_dir)) {
        fprintf(stderr, "[Server] WARN: cannot create voices dir '%s'\n", voices_dir.c_str());
    }

    tts_backend be;
    be.model_id = basename_of(talker_path);
    int n       = qt_n_speakers(q);
    for (int i = 0; i < n; i++) {
        be.voices.push_back(qt_speaker_name(q, i));
    }

    // The adapter drives the streaming pipeline. Voice resolution:
    //   1. Builtin speaker (match by name against qt_n_speakers)
    //   2. Cloned voice (load .spk + .rvq + optional .txt from VOICES_DIR)
    //   3. Fallback: pass the voice name as speaker for CustomVoice models
    //      (backward compat with the original OAI endpoint behaviour).
    be.synthesize = [q, &lang, &voices_dir](const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        struct qt_tts_params p;
        qt_tts_default_params(&p);
        p.text = req.input.c_str();
        p.lang = lang.c_str();

        // Locals that back the ref_* pointers; must outlive qt_synthesize.
        std::string         ref_text_buf;
        std::vector<float>  spk_emb;
        std::vector<int32_t> codes;
        bool                 voice_resolved = false;
        int                  n_speakers      = qt_n_speakers(q);

        if (!req.voice.empty()) {
            // 1. Builtin speaker lookup by name.
            for (int i = 0; i < n_speakers; i++) {
                if (req.voice == qt_speaker_name(q, i)) {
                    p.speaker = req.voice.c_str();
                    voice_resolved = true;
                    break;
                }
            }

            // 2. Cloned voice fallback.
            if (!voice_resolved && voice_exists(voices_dir, req.voice)) {
                int K = qt_num_codebooks(q);
                if (K <= 0) {
                    err = "codec not loaded, cannot use cloned voice";
                    return -1;
                }
                int ref_T = 0;
                if (!load_spk_file(voice_path(voices_dir, req.voice, ".spk"), spk_emb)) {
                    err = "failed to load cloned voice embedding for '" + req.voice + "'";
                    return -1;
                }
                if (!load_rvq_file(voice_path(voices_dir, req.voice, ".rvq"), K, codes, &ref_T)) {
                    err = "failed to load cloned voice codes for '" + req.voice + "'";
                    return -1;
                }
                load_txt_file(voice_path(voices_dir, req.voice, ".txt"), ref_text_buf);
                p.ref_spk_emb = spk_emb.data();
                p.ref_spk_dim = (int) spk_emb.size();
                if (!ref_text_buf.empty()) {
                    // Mode B (ICL) : use the cached RVQ codes + transcript for
                    // higher quality cloning. Skipping ref_codes when no
                    // ref_text exists avoids a pipeline assertion.
                    p.ref_codes = codes.data();
                    p.ref_T     = ref_T;
                    p.ref_text  = ref_text_buf.c_str();
                }
                // Mode A (x-vector only) : the speaker embedding alone is
                // enough for a basic clone when no ref_text was provided.
                voice_resolved = true;
            }

            // 3. Backward compat: pass unknown voice as speaker name for
            //    CustomVoice models (lets qt_synthesize surface the error).
            if (!voice_resolved && n_speakers > 0) {
                p.speaker = req.voice.c_str();
            }
        }

        if (!req.instructions.empty()) {
            p.instruct = req.instructions.c_str();
        }

        // Trampoline : the C ABI on_chunk forwards to the C++ sink.
        const tts_sink * sink_ptr = &sink;
        p.on_chunk                = [](const float * s, int ns, void * u) -> bool {
            return (*static_cast<const tts_sink *>(u))(s, ns);
        };
        p.on_chunk_user_data = (void *) sink_ptr;

        struct qt_audio out = {};
        enum qt_status  rc  = qt_synthesize(q, &p, &out);
        qt_audio_free(&out);
        if (rc != QT_STATUS_OK) {
            err = qt_last_error();
            return (int) rc;
        }
        return 0;
    };

    // -----------------------------------------------------------------------
    // HTTP server setup. We replicate tts_server_run() from the shared header
    // but register additional voice-management routes alongside the standard
    // OAI endpoints. The shared helper functions (tts_handle_speech,
    // tts_handle_models, tts_handle_health, tts_json_error, g_synth_mutex,
    // g_svr, tts_on_signal) are reused as-is.
    // -----------------------------------------------------------------------
    httplib::Server svr;
    g_svr = &svr;

    svr.set_read_timeout(60);
    svr.set_write_timeout(120);
    svr.set_payload_max_length(32 * 1024 * 1024);
    svr.set_tcp_nodelay(true);
    svr.set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    svr.set_default_headers({
        { "Access-Control-Allow-Origin", "*" }
    });
    svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    // Standard OAI endpoints (delegated to shared helpers).
    svr.Post("/v1/audio/speech",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_speech(be, req, res); });
    svr.Get("/v1/models",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_models(be, req, res); });
    svr.Get("/health", tts_handle_health);

    // Voice management endpoints.
    svr.Get("/v1/voices",
            [&be, &voices_dir](const httplib::Request & req, httplib::Response & res) {
                handle_voices_list(be, voices_dir, req, res);
            });
    svr.Post("/v1/voices/clone",
             [q, &voices_dir](const httplib::Request & req, httplib::Response & res) {
                 handle_voice_clone(q, voices_dir, req, res);
             });
    svr.Delete("/v1/voices/:voice_id",
               [&voices_dir](const httplib::Request & req, httplib::Response & res) {
                   handle_voice_delete(voices_dir, req, res);
               });

    signal(SIGINT, tts_on_signal);
    signal(SIGTERM, tts_on_signal);

    fprintf(stderr, "[Server] model %s\n", be.model_id.c_str());
    fprintf(stderr, "[Server] voices dir: %s\n", voices_dir.c_str());
    fprintf(stderr, "[Server] listening on %s:%d\n", cfg.host.c_str(), cfg.port);
    if (!svr.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", cfg.host.c_str(), cfg.port);
        qt_free(q);
        return 1;
    }

    qt_free(q);
    return 0;
}

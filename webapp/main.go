package main

import (
	"bytes"
	"embed"
	"io"
	"io/fs"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"
)

//go:embed dist/*
var staticFS embed.FS

var (
	backendURL string
	backend    *url.URL
)

// ---------- CORS ----------

func withCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

// ---------- Chat (auto-detect stream vs non-stream) ----------

func handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body failed", http.StatusBadRequest)
		return
	}

	isStream := bytes.Contains(body, []byte(`"stream":true`)) ||
		bytes.Contains(body, []byte(`"stream": true`))

	if isStream {
		doStreamChat(w, r, body)
	} else {
		doProxyChat(w, r, body)
	}
}

// streaming: manually forward and pipe SSE
func doStreamChat(w http.ResponseWriter, r *http.Request, body []byte) {
	req, _ := http.NewRequest("POST", backendURL+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if v := r.Header.Get("Authorization"); v != "" {
		req.Header.Set("Authorization", v)
	}

	resp, err := (&http.Client{Timeout: 10 * time.Minute}).Do(req)
	if err != nil {
		http.Error(w, "backend unavailable: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for _, k := range []string{"Content-Type", "X-Request-ID", "X-Process-Time", "X-Cache"} {
		if v := resp.Header.Get(k); v != "" {
			w.Header().Set(k, v)
		}
	}
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(resp.StatusCode)

	buf := make([]byte, 4096)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		if readErr != nil {
			break
		}
	}
}

// non-streaming: reverse proxy
func doProxyChat(w http.ResponseWriter, r *http.Request, body []byte) {
	req, _ := http.NewRequest("POST", backendURL+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if v := r.Header.Get("Authorization"); v != "" {
		req.Header.Set("Authorization", v)
	}

	resp, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		http.Error(w, "backend unavailable: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for _, k := range []string{"Content-Type", "X-Request-ID", "X-Process-Time", "X-Cache"} {
		if v := resp.Header.Get(k); v != "" {
			w.Header().Set(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

// ---------- Models ----------

func handleModels(w http.ResponseWriter, r *http.Request) {
	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = backend.Scheme
			req.URL.Host = backend.Host
			req.URL.Path = "/v1/models"
			req.Host = backend.Host
		},
	}
	proxy.ServeHTTP(w, r)
}

// ---------- Health ----------

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	status := "ok"
	code := http.StatusOK

	resp, err := (&http.Client{Timeout: 3 * time.Second}).Get(backendURL + "/health")
	if err != nil || resp.StatusCode != 200 {
		status = "degraded"
		code = http.StatusServiceUnavailable
	}
	if resp != nil {
		resp.Body.Close()
	}
	w.WriteHeader(code)
	w.Write([]byte(`{"status":"` + status + `","backend":"` + backendURL + `"}`))
}

// ---------- Static ----------

func staticHandler() http.Handler {
	// Strip "dist/" prefix so /assets/... maps to dist/assets/...
	distFS, err := fs.Sub(staticFS, "dist")
	if err != nil {
		log.Fatalf("failed to create sub FS: %v", err)
	}
	fileServer := http.FileServer(http.FS(distFS))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if path == "/" || path == "/index.html" {
			http.ServeFileFS(w, r, distFS, "index.html")
			return
		}
		if strings.HasSuffix(path, ".js") || strings.HasSuffix(path, ".css") {
			w.Header().Set("Cache-Control", "public, max-age=3600")
		}
		fileServer.ServeHTTP(w, r)
	})
}

// ---------- Main ----------

func main() {
	port := os.Getenv("WEB_PORT")
	if port == "" {
		port = "8080"
	}
	backendURL = os.Getenv("BACKEND_URL")
	if backendURL == "" {
		backendURL = "http://localhost:1235"
	}
	backendURL = strings.TrimRight(backendURL, "/")

	var err error
	backend, err = url.Parse(backendURL)
	if err != nil {
		log.Fatalf("invalid BACKEND_URL %q: %v", backendURL, err)
	}

	log.Printf("🚀 AI Code Service — Web Gateway")
	log.Printf("   Gateway  : http://localhost:%s", port)
	log.Printf("   Backend  : %s", backendURL)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", withCORS(handleChat))
	mux.HandleFunc("/v1/models", withCORS(handleModels))
	mux.HandleFunc("/health", withCORS(handleHealth))
	mux.Handle("/", staticHandler())

	log.Printf("==========================================")
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}

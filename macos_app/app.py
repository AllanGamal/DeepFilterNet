import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import sounddevice as sd
except Exception as e:
    sd = None
import numpy as np

from df_rt import DeepFilterRuntime


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepFilterNet2 Live")
        self.geometry("520x260")
        self.resizable(False, False)

        # Paths
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.lib_path = self._resolve_lib_path(repo_root)
        self.model_path = self._resolve_model_path(repo_root)

        # State
        self.df = None
        self.stream = None
        self.running = False
        self.lock = threading.Lock()
        self._last_proc_ms = []  # rolling list for latency display

        # UI elements
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=12, pady=(12, 6))
        self.start_btn = ttk.Button(controls, text="Start", command=self.toggle, width=12)
        self.start_btn.pack(side="left")

        # Devices row
        dev_row = ttk.Frame(self)
        dev_row.pack(fill="x", padx=12)
        ttk.Label(dev_row, text="Mikrofon:").grid(row=0, column=0, sticky="w")
        ttk.Label(dev_row, text="Högtalare:").grid(row=1, column=0, sticky="w")
        self.in_dev = tk.StringVar()
        self.out_dev = tk.StringVar()
        self.in_combo = ttk.Combobox(dev_row, textvariable=self.in_dev, width=55, state="readonly")
        self.out_combo = ttk.Combobox(dev_row, textvariable=self.out_dev, width=55, state="readonly")
        self.in_combo.grid(row=0, column=1, padx=(6, 0), pady=2, sticky="w")
        self.out_combo.grid(row=1, column=1, padx=(6, 0), pady=2, sticky="w")
        self._populate_devices()

        self.slider_label = ttk.Label(self, text="Separationsnivå (dB): 24")
        self.slider_label.pack()
        self.sep_var = tk.IntVar(value=24)
        self.slider = ttk.Scale(
            self,
            from_=0,
            to=40,
            orient="horizontal",
            command=self._on_slider,
            variable=self.sep_var,
            length=300,
        )
        self.slider.pack(pady=(0, 6))

        self.status = ttk.Label(self, text="Redo")
        self.status.pack()
        self.lat_label = ttk.Label(self, text="Latency: – ms  |  RT factor: –")
        self.lat_label.pack(pady=(4, 6))

        if sd is None:
            messagebox.showerror(
                "sounddevice saknas",
                "Installera beroenden först: pip install sounddevice numpy",
            )

    def _resolve_lib_path(self, repo_root: str) -> str:
        # Allow override via environment
        env_path = os.environ.get("DEEPFILTER_DYLIB")
        if env_path and os.path.isfile(env_path):
            return env_path
        # Cargo workspace builds to workspace-level target/ by default
        candidates = [
            os.path.join(repo_root, "target", "release", "libdf.dylib"),
            os.path.join(repo_root, "target", "release", "libdeepfilter.dylib"),
            os.path.join(repo_root, "target", "debug", "libdf.dylib"),
            os.path.join(repo_root, "target", "debug", "libdeepfilter.dylib"),
            # Fallback to per-crate target/ (non-workspace build)
            os.path.join(repo_root, "libDF", "target", "release", "libdf.dylib"),
            os.path.join(repo_root, "libDF", "target", "release", "libdeepfilter.dylib"),
            os.path.join(repo_root, "libDF", "target", "debug", "libdf.dylib"),
            os.path.join(repo_root, "libDF", "target", "debug", "libdeepfilter.dylib"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        # Not found: raise with helpful hint
        raise FileNotFoundError(
            "Hittar inte libdeepfilter/df dylib. Bygg först med:\n"
            "  cargo build -p deep_filter --release --features \"capi,default-model\"\n"
            "Förväntad fil: target/release/libdf.dylib (eller libdeepfilter.dylib).\n"
            "Alternativt ange full sökväg via env: DEEPFILTER_DYLIB=/full/path/libdf.dylib"
        )

    def _resolve_model_path(self, repo_root: str) -> str:
        # Allow override via environment
        env_model = os.environ.get("DF_MODEL_TAR")
        if env_model and os.path.isfile(env_model):
            return env_model
        # Default to DF3 which is supported by current C API
        dfn3 = os.path.join(repo_root, "models", "DeepFilterNet3_onnx.tar.gz")
        if os.path.isfile(dfn3):
            return dfn3
        # Fallback to DF2 if DF3 is not present (may not work with this runtime)
        dfn2 = os.path.join(repo_root, "models", "DeepFilterNet2_onnx.tar.gz")
        if os.path.isfile(dfn2):
            return dfn2
        raise FileNotFoundError(
            "Ingen modell hittades i models/. Förväntade DeepFilterNet3_onnx.tar.gz eller DeepFilterNet2_onnx.tar.gz.\n"
            "Du kan också sätta DF_MODEL_TAR=/full/path/model.tar.gz"
        )

    def _on_slider(self, _=None):
        val = int(float(self.sep_var.get()))
        self.slider_label.configure(text=f"Separationsnivå (dB): {val}")
        with self.lock:
            if self.df is not None:
                # Higher dB -> more allowed suppression
                self.df.set_atten_lim(float(val))

    def _populate_devices(self):
        if sd is None:
            return
        devs = sd.query_devices()
        self._dev_index_by_label = {}
        in_items = []
        out_items = []
        for idx, d in enumerate(devs):
            label = f"[{idx}] {d['name']} (in:{d['max_input_channels']} out:{d['max_output_channels']})"
            if d["max_input_channels"] > 0:
                in_items.append(label)
                self._dev_index_by_label[label] = idx
            if d["max_output_channels"] > 0:
                out_items.append(label)
                self._dev_index_by_label[label] = idx
        self.in_combo["values"] = in_items
        self.out_combo["values"] = out_items
        try:
            default_in, default_out = sd.default.device
        except Exception:
            default_in, default_out = None, None
        # Preselect defaults if available
        if default_in is not None and 0 <= default_in < len(devs):
            match = next((s for s in in_items if s.startswith(f"[{default_in}] ")), None)
            if match:
                self.in_combo.set(match)
        if default_out is not None and 0 <= default_out < len(devs):
            match = next((s for s in out_items if s.startswith(f"[{default_out}] ")), None)
            if match:
                self.out_combo.set(match)
        # Fallback to first entries
        if not self.in_combo.get() and in_items:
            self.in_combo.current(0)
        if not self.out_combo.get() and out_items:
            self.out_combo.current(0)

    def toggle(self):
        if not self.running:
            self.start()
        else:
            self.stop()

    def start(self):
        if sd is None:
            return
        try:
            self.df = DeepFilterRuntime(self.lib_path, self.model_path, atten_lim_db=float(self.sep_var.get()))
        except Exception as e:
            messagebox.showerror("Init-fel", str(e))
            return

        blocksize = self.df.frame_len
        samplerate = 48000  # DeepFilterNet operates at 48 kHz
        channels = 1

        # Resolve selected devices
        in_label = self.in_combo.get()
        out_label = self.out_combo.get()
        in_dev = self._dev_index_by_label.get(in_label, None)
        out_dev = self._dev_index_by_label.get(out_label, None)
        try:
            sd.check_input_settings(device=in_dev, channels=channels, samplerate=samplerate)
            sd.check_output_settings(device=out_dev, channels=channels, samplerate=samplerate)
        except Exception as e:
            messagebox.showerror("Enhetsfel", f"Fel på valda enheter: {e}")
            try:
                self.df.free()
            except Exception:
                pass
            self.df = None
            return

        def callback(indata, outdata, frames, time_info, status):
            if status:
                # Non-fatal warnings from PortAudio
                pass
            if frames != blocksize:
                # Drop irregular buffer sizes
                outdata.fill(0)
                return
            x = indata[:, 0].astype(np.float32, copy=False)
            t0 = time.perf_counter()
            with self.lock:
                if self.df is None:
                    outdata.fill(0)
                    return
                try:
                    _, y_ptr = self.df.process_frame(x)
                    # Copy from C buffer to numpy view
                    y = np.frombuffer(y_ptr, dtype=np.float32, count=blocksize)
                    outdata[:, 0] = y
                except Exception:
                    outdata.fill(0)
                    return
            t1 = time.perf_counter()
            proc_ms = (t1 - t0) * 1000.0
            # Keep last 30 measurements
            self._last_proc_ms.append(proc_ms)
            if len(self._last_proc_ms) > 30:
                self._last_proc_ms = self._last_proc_ms[-30:]

        try:
            self.stream = sd.Stream(
                samplerate=samplerate,
                blocksize=blocksize,
                dtype="float32",
                channels=channels,
                callback=callback,
                device=(in_dev, out_dev),
            )
            self.stream.start()
        except Exception as e:
            self.df.free()
            self.df = None
            messagebox.showerror("Audio-fel", str(e))
            return

        self.running = True
        self.start_btn.configure(text="Stop")
        self.status.configure(text=f"Kör: {samplerate/1000:.0f} kHz, block {blocksize}")
        self.after(500, self._update_metrics)

    def stop(self):
        with self.lock:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            if self.df is not None:
                try:
                    self.df.free()
                except Exception:
                    pass
                self.df = None
        self.running = False
        self.start_btn.configure(text="Start")
        self.status.configure(text="Stoppad")

    def _update_metrics(self):
        if not self.running:
            self.lat_label.configure(text="Latency: – ms  |  RT factor: –")
            return
        block_ms = (self.df.frame_len / 48000.0) * 1000.0 if self.df else 0.0
        if self._last_proc_ms:
            avg_ms = sum(self._last_proc_ms) / len(self._last_proc_ms)
            rt_factor = (avg_ms / block_ms) if block_ms > 0 else 0.0
            self.lat_label.configure(text=f"Latency: {avg_ms:.2f} ms  |  RT factor: {rt_factor:.2f}")
        else:
            self.lat_label.configure(text=f"Latency: – ms  |  RT factor: –")
        self.after(500, self._update_metrics)


if __name__ == "__main__":
    app = App()
    app.mainloop()

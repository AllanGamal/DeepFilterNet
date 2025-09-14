import ctypes
import os
from ctypes import c_char_p, c_float, c_size_t, c_void_p


class DeepFilterRuntime:
    """ctypes wrapper for libdeepfilter C API (real-time inference)."""

    def __init__(self, lib_path: str, model_path: str, atten_lim_db: float = 24.0, log_level: str | None = "INFO"):
        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"libdeepfilter dylib not found at {lib_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self._lib = ctypes.CDLL(lib_path)

        # Function signatures
        self._lib.df_create.argtypes = [c_char_p, c_float, c_char_p]
        self._lib.df_create.restype = c_void_p

        self._lib.df_get_frame_length.argtypes = [c_void_p]
        self._lib.df_get_frame_length.restype = c_size_t

        self._lib.df_process_frame.argtypes = [c_void_p, ctypes.POINTER(c_float), ctypes.POINTER(c_float)]
        self._lib.df_process_frame.restype = c_float

        self._lib.df_set_atten_lim.argtypes = [c_void_p, c_float]
        self._lib.df_set_atten_lim.restype = None

        self._lib.df_set_post_filter_beta.argtypes = [c_void_p, c_float]
        self._lib.df_set_post_filter_beta.restype = None

        self._lib.df_free.argtypes = [c_void_p]
        self._lib.df_free.restype = None

        # Optional logging hooks
        if hasattr(self._lib, "df_next_log_msg"):
            self._lib.df_next_log_msg.argtypes = [c_void_p]
            self._lib.df_next_log_msg.restype = ctypes.c_void_p
        if hasattr(self._lib, "df_free_log_msg"):
            self._lib.df_free_log_msg.argtypes = [ctypes.c_void_p]
            self._lib.df_free_log_msg.restype = None

        log_c = c_char_p(log_level.encode("utf-8")) if log_level else None
        self._state = self._lib.df_create(model_path.encode("utf-8"), c_float(atten_lim_db), log_c)
        if not self._state:
            raise RuntimeError("Failed to create DeepFilter runtime")

        self._frame_len = int(self._lib.df_get_frame_length(self._state))

    @property
    def frame_len(self) -> int:
        return self._frame_len

    def set_atten_lim(self, lim_db: float):
        self._lib.df_set_atten_lim(self._state, c_float(lim_db))

    def set_post_filter_beta(self, beta: float):
        self._lib.df_set_post_filter_beta(self._state, c_float(beta))

    def process_frame(self, in_buf) -> float:
        """Process a single frame.

        in_buf: 1D numpy array of dtype float32 with length == frame_len
        Returns: (lsnr, out_buf_ptr)
        """
        import numpy as np
        assert isinstance(in_buf, np.ndarray)
        assert len(in_buf) == self._frame_len
        assert in_buf.dtype.name == "float32"
        out = (c_float * self._frame_len)()
        snr = self._lib.df_process_frame(
            self._state,
            in_buf.ctypes.data_as(ctypes.POINTER(c_float)),
            out,
        )
        return float(snr), out

    def free(self):
        if getattr(self, "_state", None):
            self._lib.df_free(self._state)
            self._state = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


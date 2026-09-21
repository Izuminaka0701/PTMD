"""
Hash algorithms used by malware families for Dynamic API Resolution (G3).
Dùng để tính feature vector và mô phỏng pattern hash trong static analysis.
"""

from __future__ import annotations


class HashAlgorithms:
    """Common API name hashing schemes observed in MITRE-documented families."""

    @staticmethod
    def djb2(name: str, seed: int = 5381) -> int:
        h = seed
        for c in name:
            h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
        return h

    @staticmethod
    def djb2_modified(name: str, seed: int = 5381) -> int:
        """TONESHELL variant — uppercase + null terminator simulation."""
        h = seed
        for c in name.upper():
            h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
        h = ((h << 5) + h) & 0xFFFFFFFF  # extra round
        return h

    @staticmethod
    def ror13(name: str) -> int:
        """Cobalt Strike / G2 shellcode standard."""
        h = 0
        for c in name:
            h = HashAlgorithms._ror32(h, 13)
            h = (h + ord(c)) & 0xFFFFFFFF
        return h

    @staticmethod
    def splat_dropper(name: str, seed: int = 0x131313) -> int:
        """SplatDropper seed = 131313."""
        h = seed
        for c in name:
            h = ((h * 33) + ord(c)) & 0xFFFFFFFF
        return h

    @staticmethod
    def xor_encrypt(name: str, key: int = 0xAA) -> bytes:
        """CLAIMLOADER / Samurai style XOR on API name bytes."""
        return bytes([ord(c) ^ key for c in name])

    @staticmethod
    def crc32_simple(name: str) -> int:
        """AvosLocker-style checksum approximation."""
        import zlib

        return zlib.crc32(name.encode()) & 0xFFFFFFFF

    @staticmethod
    def hash_feature_vector(api_name: str) -> list[float]:
        """Compact numeric features from multiple hash schemes for one API name."""
        return [
            float(HashAlgorithms.djb2(api_name) % 65536) / 65536.0,
            float(HashAlgorithms.djb2_modified(api_name) % 65536) / 65536.0,
            float(HashAlgorithms.ror13(api_name) % 65536) / 65536.0,
            float(HashAlgorithms.splat_dropper(api_name) % 65536) / 65536.0,
            float(HashAlgorithms.crc32_simple(api_name) % 65536) / 65536.0,
            float(sum(HashAlgorithms.xor_encrypt(api_name)) % 256) / 256.0,
        ]

    @staticmethod
    def _ror32(val: int, count: int) -> int:
        val &= 0xFFFFFFFF
        return ((val >> count) | (val << (32 - count))) & 0xFFFFFFFF

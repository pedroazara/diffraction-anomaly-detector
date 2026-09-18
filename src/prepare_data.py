import hashlib
import zipfile

from src.config import RAW_DIR

ZIP_PATH = RAW_DIR / "reflex_img_1024_inter_nearest.zip"
EXPECTED_MD5 = "1b7a22ee79fe7f0690fa09f2aa9d1b12"


def check_md5(path, expected: str) -> bool:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    actual = md5.hexdigest()
    print(f"md5: {actual} (expected {expected})")
    return actual == expected


def extract():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"{ZIP_PATH} not found. Download it first.")

    print("Verifying checksum...")
    if not check_md5(ZIP_PATH, EXPECTED_MD5):
        raise ValueError("MD5 mismatch: the zip file appears to be corrupted or incomplete.")

    print(f"Extracting {ZIP_PATH.name} into {RAW_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)
    print("Done.")


if __name__ == "__main__":
    extract()

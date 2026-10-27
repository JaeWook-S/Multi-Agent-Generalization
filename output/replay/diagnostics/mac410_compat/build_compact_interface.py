"""Build a small, uncompressed MPQ containing only replay UI overrides.

MPQ table hashing/format follows StormLib (Ladislav Zezula) and is read back
with the independent mpyq reader. No map/replay data or shipped UI is edited.
"""
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mpyq


def build_compact_interface(sc2, destination):
    root = Path(__file__).resolve().parent
    template = mpyq.MPQArchive(str(Path(sc2) / "Interfaces/Streamlined.SC2Interface"))
    files = {str(path.relative_to(root / "compact_ui")).replace("/", "\\"): path.read_bytes()
             for path in (root / "compact_ui").rglob("*") if path.is_file()}
    for name, data in files.items():
        if name.lower().endswith((".sc2layout", ".sc2style")):
            ET.fromstring(data)
    # The interface type/dependency header comes from this user's installed
    # Blizzard interface. Its layouts, styles and other assets are not copied.
    files["DocumentHeader"] = template.read_file("DocumentHeader")
    files["DocumentInfo"] = template.read_file("DocumentInfo")
    files["ComponentList.SC2Components"] = b'''<?xml version="1.0" encoding="utf-8"?>
<Components>
 <DataComponent Type="info">DocumentInfo</DataComponent>
 <DataComponent Type="uiui">UI/Layout/DescIndex.SC2Layout</DataComponent>
 <DataComponent Type="font">UI/FontStyles.SC2Style</DataComponent>
 <DataComponent Type="text" Locale="enUS">GameText</DataComponent>
 <DataComponent Type="text" Locale="koKR">GameText</DataComponent>
</Components>'''
    for locale in ("enUS", "koKR"):
        files[f"{locale}.SC2Data\\LocalizedData\\GameStrings.txt"] = b"DocInfo/Name=HiSSD Compact Messages\r\n"
    files["(listfile)"] = ("\r\n".join(sorted([*files, "(listfile)"])) + "\r\n").encode("ascii")

    def encrypt_table(data, key):
        seed, carry = key, 0xEEEEEEEE
        encrypted = bytearray()
        for (word,) in struct.iter_unpack("<I", data):
            carry = (carry + template.encryption_table[0x400 + (seed & 0xFF)]) & 0xFFFFFFFF
            encrypted.extend(struct.pack("<I", (word ^ (seed + carry)) & 0xFFFFFFFF))
            seed = (((~seed << 21) + 0x11111111) | (seed >> 11)) & 0xFFFFFFFF
            carry = (word + carry + (carry << 5) + 3) & 0xFFFFFFFF
        return encrypted

    hash_count = 1
    while hash_count < len(files) * 2:
        hash_count *= 2
    hashes = [b"\xff" * 16 for _ in range(hash_count)]
    blocks = bytearray()
    payload = bytearray()
    for index, (name, data) in enumerate(files.items()):
        slot = template._hash(name, "TABLE_OFFSET") % hash_count
        while hashes[slot] != b"\xff" * 16:
            slot = (slot + 1) % hash_count
        hashes[slot] = struct.pack("<IIHHI", template._hash(name, "HASH_A"),
                                   template._hash(name, "HASH_B"), 0, 0, index)
        blocks.extend(struct.pack("<IIII", 32 + len(payload), len(data), len(data), 0x81000000))
        payload.extend(data)
    hash_offset = 32 + len(payload)
    block_offset = hash_offset + hash_count * 16
    archive_size = block_offset + len(blocks)
    header = struct.pack("<4sIIHHIIII", b"MPQ\x1a", 32, archive_size, 0, 3,
                         hash_offset, block_offset, hash_count, len(files))
    destination = Path(destination)
    destination.write_bytes(header + payload
                            + encrypt_table(b"".join(hashes), template._hash("(hash table)", "TABLE"))
                            + encrypt_table(blocks, template._hash("(block table)", "TABLE")))
    check = mpyq.MPQArchive(str(destination))
    for name, data in files.items():
        if check.read_file(name) != data:
            raise RuntimeError(f"Interface archive verification failed: {name}")
    return destination


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "HiSSDCompact.SC2Interface"
    build_compact_interface("/Applications/StarCraft II", target)
    print(target, target.stat().st_size)

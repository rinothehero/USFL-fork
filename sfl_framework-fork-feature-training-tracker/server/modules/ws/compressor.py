import time

import brotli


def compress(data: bytes) -> bytes:
    # start_time = time.time()
    # original_size_mb = len(data) / (1024 * 1024)

    # while True:
    #     compressed_data = brotli.compress(data, quality=1)

    #     try:
    #         decompressed_data = brotli.decompress(compressed_data)
    #         if decompressed_data == data:
    #             break
    #     except Exception as e:
    #         print("Compression verification failed. Retrying...")
    #         continue

    # end_time = time.time()
    # compressed_size_mb = len(compressed_data) / (1024 * 1024)

    # print(f"Compression time: {end_time - start_time:.2f} seconds")
    # print(f"Original size: {original_size_mb:.2f} MB")
    # print(f"Compressed size: {compressed_size_mb:.2f} MB")
    # print(
    #     f"Compression ratio: {(1 - compressed_size_mb / original_size_mb) * 100:.2f}%"
    # )

    # return compressed_data
    return data


def decompress(data: bytes) -> bytes:
    # start_time = time.time()
    # decompressed_data = brotli.decompress(data)
    # end_time = time.time()

    # print(f"Decompression time: {end_time - start_time:.2f} seconds")
    # return decompressed_data
    return data

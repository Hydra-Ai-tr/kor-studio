// Kor Studio - image and sound files, written from scratch: PNG (read and
// write), BMP and PPM/PGM (read), WAV (read and write).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace kor {

// 8-bit RGB pixels, row by row.
struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
};

// Reads .png, .bmp, .ppm and .pgm files. Transparent pixels are blended onto white.
bool load_image(const std::string& path, Image& img, std::string* err);
bool save_png(const std::string& path, const Image& img, std::string* err);
// Crops the middle square of the image and scales it to size x size.
Image square_thumbnail(const Image& img, int size);
bool is_image_file(const std::string& path);

// Mono samples in [-1, 1].
struct Sound {
    int rate = 0;
    std::vector<float> samples;
};

// Reads PCM (8/16/24/32-bit) and 32-bit float WAV files; channels are mixed to mono.
bool load_wav(const std::string& path, Sound& snd, std::string* err);
bool save_wav(const std::string& path, const Sound& snd, std::string* err);
std::vector<float> resample(const std::vector<float>& x, int from_rate, int to_rate);
bool is_wav_file(const std::string& path);

// 8-bit mu-law companding (256 levels), as used by telephone audio.
int mulaw_encode(float x);
float mulaw_decode(int level);

// Demo data: a folder of small pictures (coloured shapes on gradients) and a
// folder of short melodies, to try image and sound models without own files.
bool write_demo_images(const std::string& dir, int count = 1000, std::string* err = nullptr);
bool write_demo_sounds(const std::string& dir, int count = 60, std::string* err = nullptr);

// zlib streams (used by PNG).
bool zlib_inflate(const uint8_t* data, size_t n, std::vector<uint8_t>& out);
uint32_t crc32(const uint8_t* data, size_t n, uint32_t crc = 0);

}  // namespace kor

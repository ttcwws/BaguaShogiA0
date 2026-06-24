#pragma once
#include "rand.h"
Hash128& Hash128::operator^=(const Hash128& b) { return *this = *this ^ b; }
Hash128 operator^(const Hash128& a, const Hash128& b)
{
    return Hash128(a.hash0 ^ b.hash0, a.hash1 ^ b.hash1);
}
bool operator==(const Hash128& a, const Hash128& b)
{
    return a.hash0 == b.hash0 && a.hash1 == b.hash1;
}
bool operator!=(const Hash128& a, const Hash128& b)
{
    return a.hash0 != b.hash0 || a.hash1 != b.hash1;
}
uint64_t StringHash(const char* s)
{
    unsigned t = strlen(s);
    uint64_t ret = 0, p = 1;
    for (int i = 0; i < t; ++i)
    {
        ret = (ret + (s[i] * p) % HASH_MODULO) % HASH_MODULO;
        p = (p * 256) % HASH_MODULO;
    }
    return ret;
}
void Rand::init(const uint64_t& seed) { e = std::mt19937_64(seed); }
void Rand::init(const char* s) { init(StringHash(s)); }
uint64_t Rand::nextUInt() { return rand(e); }
std::vector<float> dirichlet(int k, float a) {
    static thread_local std::gamma_distribution<float> gamma(a);
    static thread_local std::mt19937 rd(std::random_device{}());
    std::vector<float> y;
    y.resize(k);
    float sum = 0;
    while (sum < FLT_EPSILON)
    {
        for (int i = 0; i < k; ++i) {
            y[i] = gamma(rd);
            sum += y[i];
        }
    }
    for (float& i : y)i /= sum;
    return y;
}

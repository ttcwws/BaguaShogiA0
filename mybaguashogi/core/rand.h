#pragma once
#include <cstring>
#include <cstdint>
#include <random>
#include <iostream>

constexpr uint64_t HASH_MODULO = 4294967291ull;
class Hash128
{
public:
    uint64_t hash0, hash1;
    Hash128(uint64_t h0, uint64_t h1) :hash0(h0), hash1(h1) {}
    Hash128() :hash0(0), hash1(0) {}
    Hash128(const Hash128&) = default;
    Hash128& operator^=(const Hash128& b);
};
Hash128 operator^(const Hash128& a, const Hash128& b);
bool operator!=(const Hash128& a, const Hash128& b);
uint64_t StringHash(const char* s);
class Rand
{
    std::mt19937_64 e;
    std::uniform_int_distribution<uint64_t> rand;
public:
    Rand() :e(0), rand(0) {}
    Rand(const uint64_t& seed) :e(seed), rand(0) {}
    Rand(const char* s) :e(StringHash(s)), rand(0) {}
    uint64_t nextUInt();
    void init(const uint64_t& seed);
    void init(const char* s);
};
template <class T>
size_t chooseWithProbability(const std::vector<T>& probs, uint64_t seed_off = 0)
{
    std::random_device e;
    std::discrete_distribution<> d(probs.begin(), probs.end());
    return d(e);
}
std::vector<float> dirichlet(int k, float a);

#pragma once

#include "cpam/cpam.h"
#include "pam/pam.h"

template<typename Type>
struct addm {
  using T = Type;
  static T identity() { return 0; }
  static T add(T a, T b) { return a + b; }
};

template<typename Type>
struct minm {
  using T = Type;
  static T identity() { return std::numeric_limits<Type>::max(); }
  static T add(T a, T b) { return std::min(a, b); }
};

template<typename key_type>
struct basic_entry {
  using key_t = key_type;
  using val_t = key_type;
  static inline bool comp(key_t a, key_t b) { return a < b; }
};

template<typename key_type, typename val_type>
struct aug_entry {
  using key_t = key_type;
  using val_t = val_type;
  using aug_t = std::pair<val_t, key_t>;

  static inline bool comp(key_t a, key_t b) { return a < b; }
  // The following three functions specify the augmentation:
  // get_empty() is the identity value
  // from_entry(...) maps a (K,V) pair to an augmented value
  // combine(...) is the associative augmentation fn.
  static aug_t get_empty() {
    return std::make_pair(std::numeric_limits<val_t>::max(), std::numeric_limits<key_t>::max());
  }
  static aug_t from_entry(key_t k, val_t v) { return std::make_pair(v, k); }
  static aug_t combine(aug_t a, aug_t b) { return std::min(a, b); }
};

// using integer_map = cpam::pam_map<basic_entry, 32>;
// using integer_aug_map = cpam::aug_map<aug_entry, 32>;
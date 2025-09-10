#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

// Helper to map a file for reading, and return (void*, size_t).
inline std::pair<char*, size_t> mmap_file(const char* filename) {
  int fd = open(filename, O_RDONLY);
  if (fd == -1) throw std::runtime_error("Failed to open file for mmap");

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    throw std::runtime_error("Failed to stat file for mmap");
  }
  size_t len = sb.st_size;
  char* addr = static_cast<char*>(mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0));
  close(fd);
  if (addr == MAP_FAILED) throw std::runtime_error("mmap failed");
  return {addr, len};
}
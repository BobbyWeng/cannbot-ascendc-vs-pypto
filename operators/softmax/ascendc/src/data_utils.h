#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

inline bool ReadFile(const std::string &filePath, size_t bufferSize, void *buffer, size_t bufferLen) {
    if (buffer == nullptr || bufferSize > bufferLen) return false;
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end); size_t fileSize = file.tellg(); file.seekg(0, std::ios::beg);
    if (fileSize != bufferSize) return false;
    file.read(static_cast<char *>(buffer), bufferSize);
    if (!file) { file.close(); return false; }
    file.close(); return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size) {
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) return false;
    ssize_t writeSize = write(fd, buffer, size);
    close(fd);
    return static_cast<size_t>(writeSize) == size;
}
#endif

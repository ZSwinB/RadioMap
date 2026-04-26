#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    char c;
    int i;
    double d;
} Data;

// 手动序列化到 13 字节缓冲区
void serialize(uint8_t buf[13], const Data *src) {
    memcpy(buf,     &src->c, 1);
    memcpy(buf + 1, &src->i, 4);
    memcpy(buf + 5, &src->d, 8);
}

// 手动从 13 字节缓冲区反序列化
void deserialize(Data *dst, const uint8_t buf[13]) {
    memcpy(&dst->c, buf,     1);
    memcpy(&dst->i, buf + 1, 4);
    memcpy(&dst->d, buf + 5, 8);
}

int main() {
    Data x = { 'A', 123, 3.14 };
    uint8_t buf[13];
    Data y;

    serialize(buf, &x);
    deserialize(&y, buf);

    printf("buf size = %zu\n", sizeof(buf));
    printf("y.c = %c\n", y.c);
    printf("y.i = %d\n", y.i);
    printf("y.d = %.2f\n", y.d);

    return 0;
}
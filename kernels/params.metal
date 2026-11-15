// WBITS is the bit width of the model's main weights. The host puts a
// #define in front of the source when the model is not 5-bit.
#ifndef WBITS
#define WBITS 5
#endif

// What changes from one token to the next. The CPU writes it before each
// command buffer; kernels that need it take it as one more buffer.
// Must match plan::TokenParams.
struct TokenParams {
    uint token;     // token id being processed
    uint position;  // its position in the sequence, from 0
    uint kv_len;    // entries in the KV cache after this token: position + 1
    uint pad;
};

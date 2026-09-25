// F-13 regression: with the persistent fp16 weight cache enabled, the fused QKV
// activation scratch MUST exist, otherwise the forward silently falls back to
// three separate projections per layer.
#include "model/model.h"
#include <iostream>
using namespace gai;
static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){ std::cerr << "FAIL: " << msg << "\n"; ++failures; } } while(0)

static ModelConfig cfg_of(bool qk) {
  ModelConfig c; c.vocab_size=32; c.hidden_size=32; c.num_layers=2; c.num_heads=4;
  c.num_kv_heads=2; c.intermediate_size=64; c.max_seq_len=32; c.use_qk_norm=qk; return c;
}
int main(){
  for (int qk=0; qk<2; ++qk) {
    ModelConfig cfg = cfg_of(qk!=0);
    // cache OFF -> no fused qkv scratch by design
    { Model m(cfg, Device::CPU); m.init_weights(1);
      Activations a = m.make_activations(1, 8, true, 1);
      CHECK(!a.qkv.defined(), "no fused qkv scratch when the fp16 cache is off"); }
    // cache ON -> fused qkv scratch present (this is what F-13 was losing)
    { Model m(cfg, Device::CPU); m.init_weights(1);
      m.enable_fp16_weight_cache(true);
      CHECK(m.fp16_weight_cache_enabled(), "fp16 weight cache reports enabled");
      CHECK(m.fused_qkv_ptr(0) != nullptr, "per-layer fused qkv fp16 cache exists");
      CHECK(m.fp16_weight_cache_bytes() > 0, "fp16 cache has a non-zero footprint");
      Activations a = m.make_activations(1, 8, true, 1);
      CHECK(a.qkv.defined(), "F-13: fused qkv scratch is allocated when the cache is on");
      // and the forward must run through it
      m.enable_grad(true);
      std::vector<i32> ids(8), tgt(8, -100);
      for (int i=0;i<8;++i){ ids[i]=i+1; if(i+1<8) tgt[i]=ids[i+1]; }
      double loss = m.forward_backward(ids.data(), tgt.data(), 1, 8, a);
      CHECK(loss == loss, "F-13: forward_backward runs on the fused qkv path (finite loss)");
    }
  }
  if (failures==0){ std::cout << "test_fp16_cache_order: ALL PASS\n"; return 0; }
  std::cerr << "test_fp16_cache_order: " << failures << " FAILURES\n"; return 1;
}

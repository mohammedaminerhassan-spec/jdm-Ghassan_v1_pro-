# 🔍 تقرير التدقيق العميق — Ghassan v1 Pro على Kaggle 4×T4
**التاريخ:** 2026-09-26 · **النطاق:** HEAD الحالي (469fa8f9ce) · **الهدف:** تشخيص الأسباب التي تجعل الموديل لا يكمل التدريب على Kaggle 4×T4 — بدون lag، بدون crash، وبدون أخطاء — مع تحليل عميق للكود والإعدادات.

---

## ❓ السؤال البحثي
لماذا لا تكتمل تدريبات Ghassan v1 Pro على جلسات Kaggle T4 (فردي و4×T4)؟ وأين هي الأخطاء والتحذيرات الحقيقية المتبقية في الكود الحالي التي تمنع تدريبًا سريعًا ومكتملًا على الـ datasets؟

---

## 🎯 الملخص التنفيذي (مرتّب حسب الخطورة)

1. **[P0] باگ قاتل في `save_async`: بوابة "snapshot مزدوج" تخنق أي run طويل على ذاكرة Kaggle.** الميزانية = 35% من RAM الفيزيائي (~11.0 GB على Kaggle)، بينما شرط النجاح هو `2 × حجم snapshot ≤ الميزانية`. الموديل 480M يحتاج 11.5 GB ← **يفشل**، والموديل 1B يحتاج 24.6 GB ← **يفشل على أي جهاز أقل من 70 GB RAM**. اليموت في أول `save_every` (الخطوة 500) — أي بعد ساعات من التدريب.
2. **[P0] `configs/sft_en_4xt4.yaml` ناقص/مخربك مع `en_pro.yaml`:** ينقصه `use_qk_norm: true` (والافتراضي false) فماشي مثل القاعدة المكتوبة في `sft_pro_v1.yaml` نفسها ("model.* block byte-identical"). النتيجة: فشل تحميل `pretrained_checkpoint` فورًا برسالة `checkpoint tensor mismatch: ...qk_qnorm...` — **مسار 4×T4 SFT كامل غير قابل للتشغيل**.
3. **[P1] لا يوجد أي test يفرض تطابق block الـ model بين config التدريب وconfig الـ SFT** — وهذا بالضبط ما سمح للباگ رقم 2 بالوصول إلى الـ HEAD رغم 26 commit إصلاح في نفس اليوم.
4. **[P1] `setup.sh` لا يتحقق من تثبيت NCCL** — إن غاب `libnccl-dev` عن صورة Kaggle يتحول البناء بصمت إلى single-GPU، ثم يفشل `train_4xt4.sh` في وقت التشغيل.
5. **[P2] `train_4xt4.sh` لا يفعّل بوابة `--output-budget-mb`** (الافتراضي 0 = معطّلة) بينما `train_1b.sh` يحرّضها على 19456 — نفس خطر "Save Version" الذي أصلحه الـ commit 5567fda9 لكنه غير مفعّل في مسار 4×T4.
6. **[P2] ملاحظات توثيق:** تعليق `train_4xt4.sh` يحيل على `configs/ultra_1b.yaml` غير الموجود؛ والـ snapshot يتضمن الـ gradients في ملفات الـ checkpoint (12 بايت/بارامتر بدل 8) وهذا يهدر نصف حصة الـ 19.5 GB saved-output بلا فائدة.

---

## 🧪 المنهجية
- قراءة كاملة أو شبه كاملة لأكثر من **70 ملفًا**: كل الـ configs، كل سكريبتات `kaggle/`، كل الـ headers، `training/distributed.cpp`، `dataloader.cpp`، `optimizer.cpp`، `scheduler.h`، `core/config.cpp`، `cuda/cuda_utils.cu`، `cuda/attention.cu`، `core/ops_moe.cpp`، `tools/train_main.cpp`، و6 من ملفات الـ tests.
- الملفات الكبيرة جدًا (`trainer.cpp` 113KB، `model.cpp` 94KB، `moe.cu` 53KB) قُرئ رأسها كاملًا (أول ~32KB)، ثم فُحصت ذيولها عبر **GitHub code search** المستهدف لكل دالة حرجة (`run_pretrain`, `save_async`, `sync_gradients`, `sync_eval_best`, `opt_step`, `check_ckpt_health`, `wait_for_save`, ...) وعبر **diffs الـ commits** الحديثة (111c842, 2f9ee46).
- تدقيق رقمي مستقل لكل معادلة حراسة (RAM/disk/quota/VRAM) على مواصفات Kaggle الحقيقية (4×T4, 16GB VRAM لكل واحد، ~29GB RAM، 4 أنوية CPU).
- **حدود التغطية:** لم يُترجم كل سطر من أذيال الملفات العملاقة (~30% من `trainer.cpp` و`model.cpp` لم يُقرأ سطرًا بسطر) — لكن كل المسارات الحرجة للحفظ/التزامن/الخسارة/الإشارات تم التحقق منها عبر search + diffs. لم يتم تشغيل تجارب فعلية على GPU (لا يتوفر CUDA هنا) — الاستنتاجات ساكنة (static) لكن مبنية على أرقام القياس المذكورة في commits المشروع نفسها.

---

## 🔬 النتائج التفصيلية

### 🔴 P0-1: بوابة RAM ديال checkpoints كاتقتل أي run طويل على Kaggle
**الملفات:** `training/trainer.cpp` (`save_async`, `ckpt_ram_budget`) — أُدخلت في commit `111c842a` ("guard the two ways a long run actually dies").

السلسلة المنطقية كما هي في الكود:
1. `ckpt_ram_budget() = min(35% × RAM الفيزيائي, 32GB)` مع أرضية 8GB.
2. في `save_async`: `need_two = snapshot->bytes() * 2; if (need_two > budget) GAI_FAIL("host RAM cannot checkpoint this model: ... Backpressure would never clear ...")`.
3. المشروع نفسه قاس أن snapshot = **12.0002 بايت/بارامتر** (نص الـ commit: "the 480M model queues 5.76 GB snapshots").

**الحساب على Kaggle (RAM ≈ 31.5GB فيزيائي → ميزانية ≈ 11.0GB):**

| الوصفة | حجم snapshot | need_two | الميزانية | النتيجة |
|---|---|---|---|---|
| en_pro 480M (Lion) | 5.76 GB | 11.52 GB | ~11.0 GB | ❌ فشل عند أول حفظ |
| pro_v1 / t4_1b 1B (Lion) | ~12.3 GB | ~24.6 GB | ~11.0 GB | ❌ فشل دائمًا على Kaggle |
| sft_en_4xt4 (480M) | 5.76 GB | 11.52 GB | ~11.0 GB | ❌ فشل عند أول best-save |

التناقض الداخلي: pre-flight الـ constructor يستعمل شرطًا آخر (`need = 2×snapshot + 512MB ≤ RAM الكلي` ≈ 12.0GB على Kaggle ✓ ينجح)، فتمر الجلسة من الفحص الابتدائي ثم **تموت بعد ساعات** عند الخطوة `save_every=500` — وهذا نمط الفشل الذي وصفته بالضبط: "التدريب يمشي مزيان ثم ما كايكملش".

**الإصلاحات المقترحة (بأي ترفيبة واحدة تكفي، والأفضل أول اثنين):**
- **(أ)** لا تحفظ الـ gradients في الـ snapshot إطلاقًا (هي غير ضرورية للاستئناف): 12 → 8 بايت/بارامتر. en_pro: need_two تصبح 7.7GB ✓. هذا أيضًا يصغّر ملفات `.ckpt` بنسبة ~33% ويخفف ضغط حصة 19.5GB والـ ENOSPC.
- **(ب)** عندما `need_two > budget`: **انتظر (drain) بدل الفشل** — نداء `wait_for_save()` ثم أعد المحاولة؛ توقف بضع ثوانٍ كل 500 خطوة أفضل موت الجلسة كلها.
- **(ج)** ارفع الحصة للجلسات المخصصة (Kaggle لا يشاركه شيء آخر): 35% → 60% مع نفس العدّ الدقيق `unique_snapshot_bytes`.
- **(د)** وحّد شرط الـ constructor مع شرط `save_async` حتى لا يتناقضا.

### 🔴 P0-2: `configs/sft_en_4xt4.yaml` يكسر القاعدة الذهبية للتطابق مع en_pro
المقارنة سطرًا بسطر مع `configs/en_pro.yaml` (القاعدة المعلنة في `configs/sft_pro_v1.yaml`: *"Keep this file's model.* block byte-identical to pro_v1.yaml or resume fails the architecture gate"*):

| المفتاح | en_pro.yaml (pretrain) | sft_en_4xt4.yaml | الافتراضي في `model.h` | الأثر |
|---|---|---|---|---|
| `use_qk_norm` | `true` | **غائب** | `false` | ❌ الموديل يُبنى بدون `qk_qnorm/qk_knorm` (26×2 بارامترات ناقصة) → تحميل `en_pro/last.ckpt` يفشل: `checkpoint tensor mismatch` |
| `z_loss_scale` | `0.0001` | **غائب** | `0.0` | سلوك router يختلف عن التدريب الأساسي (warn-only لكنه انحراف recipe) |
| `moe_aux_free` | `false` | غائب | `false` | ✓ متطابق بالصدفة |
| `rope_scale/yarn_*/sliding_window/rope_type` | معرّفة | غائبة | نفس القيم | ✓ (لكن هشّ — أي تعديل مستقبلي في en_pro يكسر الـ SFT بصمت) |
| `fp16_weight_cache` (training) | `true` في sft_en_pro | **غائب** | `false` | خسارة أداء (مسار QKV غير مدمج) في 4×T4 |

**الإصلاح المقترح:** انسخ block الـ `model:` من `sft_en_pro.yaml` حرفيًا (وقد أضف `fp16_weight_cache: true`).

### 🟠 P1-1: غياب test لفرض تطابق أزواج الـ configs
`tests/test_configs.cpp` يفحص parsing والـ globs فقط. لا يوجد شيء يمنع تكرار نفس الخطأ (وهو باگ وصل للـ HEAD رغم كل الـ hardening). **المقترح:** test يقرأ أزواج `(pretrain, sft)` ويقارن `model.*` حرفيًا: `(en_pro, sft_en_pro)`, `(en_pro, sft_en_4xt4)`, `(pro_v1, sft_pro_v1)`. أو — أعمق — بوابة runtime في `Trainer` عند `stage=sft`: قارن math-fields من metadata الـ checkpoint مع config الموديل قبل التحميل وقل رسالة قابلة للفهم ("SFT config ينقصه use_qk_norm: true").

### 🟠 P1-2: `setup.sh` لا يضمن NCCL لسكريبت 4×T4
`CMakeLists.txt` (سطر 135-147) يبحث عن `nccl.h` ويطبع فقط `[ghassan-ai] NCCL: not found -> single-GPU` — تحذير بنيّ يمر بصمت. `kaggle/setup.sh` لا يثبّت `libnccl2/libnccl-dev` ولا يفحص نتيجة الاكتشاف. إن غابت الحزمة عن صورة Kaggle: البناء ينجح "single-GPU only" ثم `train_4xt4.sh` يموت في أول `ncclCommInitRank`... لا — أسوأ: `distributed.cpp` يفشل مباشرة بـ "Multi-GPU requested but NCCL not available" بعد ما تكون قد أضعت وقت البناء كاملًا. **المقترح:** بعد cmake: تحقق من السطر في output، وعند 4×T4 أضف `apt-get install -y -qq libnccl2 libnccl-dev` أو افشل بصوت عالٍ.

### 🟡 P2-1: بوابة حصة الـ saved-output غير مفعّلة في 4×T4
الـ commit 5567fda9 أضاف `--output-budget-mb` (19456 = رقم Kaggle الحقيقي) وحرّضه في `train_1b.sh` و`cell8_pilot.sh` — لكن `train_4xt4.sh` يطلق `gai_train` بدونها (الافتراضي 0 = off). نفس فئة الخطر التي وثّقها المشروع نفسه ("training finishes, then Save Version refuses"). **المقترح:** مرّر `--output-budget-mb 19456` في `train_4xt4.sh` (أو اجعل الافتراضي يعمل عندما يُكتشف `/kaggle/working`).

### 🟡 P2-2: ملاحظات صغيرة
- تعليق `train_4xt4.sh` يحيل إلى `configs/ultra_1b.yaml` (غير موجود؛ المقصود `pro_v1.yaml`).
- الـ snapshot يحمل gradients (12B/param) — تضخيم غير ضروري لملفات checkpoint ضد حصة القرص (انظر إصلاح P0-1أ).
- `sft_en_4xt4.yaml`: `warmup_steps: 100` مع `epochs: 3` — إذا كانت الشردات صغيرة بحيث `planned_total ≤ 100` سيفشل الـ trainer بفحص warmup (احذر عند تجارب صغيرة).
- 4 ranks × (main + prefetch + ckpt-writer) = ~12 thread على 4 أنوية — مقبول لأنها I/O-bound، لكن راقب `nvidia-smi` أول 10 خطوات.

---

## ✅ ما تم التحقق منه ووجدته سليمًا (لمنع إعادة اختراع العجلة)
هذه المناطق دُققت ووجدت **صحيحة ومحصّنة** — لا تعِد كتابتها:
- **NCCL/DDP:** staging للمؤشرات host عبر device قبل collective (48504f63)، ملف rendezvous يُحذف قبل كل spawn (fix الـ hang)، barrier على device، F-01 بروتوكول `sync_eval_best` يُرجع القرار المتفق عليه، بذور sampler مملّحة بالـ rank (لا تكرار بيانات)، درقة `WORLD_SIZE` قابلة للتجاوز.
- **loss scaling:** clamp عند 16384 (kLossScaleMax) لأن |dlogits| يبلغ الـ scale نفسه — و`loss_scale_init: 8192` المقيس مدرج في كل الـ yamls.
- **Hot path:** لا `cudaMalloc/cudaFree` في الحلقات (workspaces مسبقة الحجم عبر `Model::workspace_plan`)، لا `cudaDeviceSynchronize` إلا في مسارات باردة، MoE counters على device بلا roundtrip، prefetch thread + async ckpt writer = GPU لا ينتظر I/O (لا "lag" بنيوي هنا).
- **kernels:** grid-stride loops في moe.cu (2f9ee46) أصلحت فئة crash كاملة؛ guards الـ shared-memory 48KB و`head_dim` و`K∈[1,8]`/`ne≤64` موجودة.
- **OOM/ENOSPC/signals:** guards VRAM للوصفة 1B (B=1/T=512 إجباري)، SIGTERM يوقف عند حدود خطوة ويحفظ `last.ckpt`، `TmpGuard` ينظف `.tmp/.bak`، نُشر الـ checkpoint atomic.
- **dataloader:** فحوص OOB، thread-local fd cache، u16/u32 + masks مُتحقق منها.
- **optimizer:** `validate_moment_table` قبل أي kernel، تخطي التحديث عند non-finite مع عدّ `skipped_steps`، Muon/grad_scale invariance مثبتة بـ test.
- **setup.sh:** JOBS≤2 لحماية RAM البناء، ccache، بوابة GPU، Blackwell check.
- **configs الأخرى:** `sft_en_pro.yaml` و`sft_pro_v1.yaml` و`pro_auxfree.yaml` كلها متطابقة معماريًا مع قواعدها ✓ — الخلل محصور في `sft_en_4xt4.yaml`.

---

## 📚 ملاحظات المصادر

| المصدر | المصداقية | آخر تحديث |
|---|---|---|
| [training/trainer.cpp (HEAD)](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-/blob/main/training/trainer.cpp) — `ckpt_ram_budget`/`save_async` | 5/5 | 2026-09-26 |
| [configs/sft_en_4xt4.yaml](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-/blob/main/configs/sft_en_4xt4.yaml) vs [configs/en_pro.yaml](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-/blob/main/configs/en_pro.yaml) | 5/5 | 2026-09-26 |
| [commit 111c842a](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-/commit/111c842a08b2f26441d7f2db4b9e05453b06f70a) — نص "12 bytes/param, 5.76GB snapshot" و"35% RAM" | 5/5 | 2026-09-26 |
| [training/checkpoint.cpp](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-/blob/main/training/checkpoint.cpp) — `checkpoint tensor mismatch` on missing param | 5/5 | 2026-09-26 |
| [model/model.h](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-/blob/main/model/model.h) — الافتراضيات (`use_qk_norm=false`, `z_loss_scale=0`) | 5/5 | 2026-09-26 |
| [CMakeLists.txt](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-/blob/main/CMakeLists.txt) — اكتشاف NCCL الاختياري | 5/5 | 2026-09-26 |
| [Kaggle: GPU Notebooks 4 CPUs & 29GB RAM](https://www.kaggle.com/discussions/product-feedback/448251) | 4/5 | - |
| [Reddit r/LocalLLaMA: Kaggle T4 29GB](https://www.reddit.com/r/LocalLLaMA/comments/17bhwtj/kaggle_upgraded_their_free_tier_to_t4_with_29gb/) | 3/5 | - |

**تعارضات وتحفظات:** الرقم الدقيق لـ RAM الفيزيائي على جلسة T4×4 يختلف قليلًا (29–31.5GB حسب العرض) — لكن حتى بأقصى تقدير (ميزانية 11.5GB) فالـ 480M على الحرف الرفيع جدًا و1B مستحيل؛ الاستنتاج لا يتغير. حساب snapshot = 12B/param مأخوذ من قياس المشروع نفسه وليس من إعادة قياس مستقلة.

---

## ❓ أسئلة مفتوحة
1. هل snapshot يشمل فعلًا الـ gradients (يفسّر 12B/param) أم أنّ 4B زائدة من مصدر آخر؟ يحتاج قياسًا واحدًا بـ `test_snapshot_memory` على ملف حقيقي.
2. هل صورة Kaggle الحالية تحتوي `nccl.h` افتراضيًا؟ (لم أستطع التحقق من داخل الصورة — P1-2 يغطي الحالتين).
3. سلوك `wait_for_save()` داخل `save_async` عند القفل العكسي (deadlock بين backpressure وdrain) يحتاج test متعدد الخيوط قبل اعتماد إصلاح (ب).

## 🚀 التوصيات (بالترتيب)
1. **اليوم:** أصلح `configs/sft_en_4xt4.yaml` (block model مطابق لـ en_pro + `fp16_weight_cache: true`).
2. **اليوم:** أصلح بوابة `save_async` (إصلاح أ + ب أعلاه) — هذا وحده هو الفارق بين "run يكمل" و"run يموت عند الخطوة 500".
3. **أضف test تطابق أزواج configs** (P1-1) لكي لا يتكرر P0-2 أبدًا.
4. **فعّل `--output-budget-mb 19456` في `train_4xt4.sh`** وافحص NCCL في `setup.sh`.
5. **أعد تشغيل `kaggle/cell8_pilot.sh`** بعد الإصلاحات ثم smoke 2-rank (`CUDA_LAUNCH_BLOCKING=1`) قبل إطلاق الجلسة الكاملة 4×T4.
6. ثم أطلق: pretrain بـ `t4_1b.yaml` عبر `train_4xt4.sh`، بعدها SFT بـ `sft_en_4xt4.yaml` المصحح — مع `resume_mode: exact` للإنتاج.

*التقرير ساكن (static audit) مبني على HEAD 469fa8f9ce؛ كل أرقام القياس (5.76GB، 12B/param، 8192 loss-scale) منقولة من وثائق commits المشروع نفسه.*

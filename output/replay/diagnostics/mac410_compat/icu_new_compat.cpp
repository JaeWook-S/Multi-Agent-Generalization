// Process-local workaround experiment for the observed SC2 4.10 startup cycle:
// SC2 initializer -> NSRegularExpression -> ICU -> SC2 operator new -> initializer.
// v2: ONLY the ICU image's allocator imports are rebound. Do not globally
// interpose C++ new/delete: SC2 has internal direct delete calls that bypass
// dyld interposition and cannot safely receive fallback allocations.
// ICU allocations made inside uregex_open use malloc; their deletes are
// tracked by the ICU-scoped delete wrappers.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <new>
#include <pthread.h>
#include <unistd.h>
#include <mach-o/loader.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <cstring>

extern "C" void *uregex_open(const uint16_t *, int32_t, uint32_t, void *, int32_t *);

namespace {
using New = void *(*)(size_t);
using Delete = void (*)(void *);
using SizedDelete = void (*)(void *, size_t);
using Regex = void *(*)(const uint16_t *, int32_t, uint32_t, void *, int32_t *);
struct InterposePair { const void *replacement; const void *replacee; };
using DynamicInterpose = void (*)(const mach_header *, const InterposePair *, size_t);
New original_new = nullptr;
New original_new_array = nullptr;
Delete original_delete = nullptr;
Delete original_delete_array = nullptr;
SizedDelete original_sized_delete = nullptr;
SizedDelete original_sized_delete_array = nullptr;
Regex original_regex = nullptr;
DynamicInterpose dynamic_interpose = nullptr;
const mach_header *icu_header = nullptr;
bool icu_rebound;
__thread unsigned regex_depth;
constexpr size_t slot_count = 1u << 20;
void *owned[slot_count];
pthread_mutex_t owned_lock = PTHREAD_MUTEX_INITIALIZER;
void *const tombstone = reinterpret_cast<void *>(uintptr_t(1));

[[noreturn]] void fail(const char *message) {
  write(STDERR_FILENO, message, __builtin_strlen(message));
  abort();
}

size_t start_slot(void *p) {
  return ((reinterpret_cast<uintptr_t>(p) >> 4) * uintptr_t(11400714819323198485ull)) & (slot_count - 1);
}

void remember(void *p) {
  pthread_mutex_lock(&owned_lock);
  size_t i = start_slot(p);
  for (size_t n = 0; n < slot_count; ++n, i = (i + 1) & (slot_count - 1)) {
    if (!owned[i] || owned[i] == tombstone) {
      owned[i] = p;
      pthread_mutex_unlock(&owned_lock);
      return;
    }
  }
  pthread_mutex_unlock(&owned_lock);
  fail("[sc2-icu-compat] pointer tracking capacity exceeded\n");
}

bool forget(void *p) {
  pthread_mutex_lock(&owned_lock);
  size_t i = start_slot(p);
  for (size_t n = 0; n < slot_count; ++n, i = (i + 1) & (slot_count - 1)) {
    if (!owned[i]) break;
    if (owned[i] == p) {
      owned[i] = tombstone;
      pthread_mutex_unlock(&owned_lock);
      return true;
    }
  }
  pthread_mutex_unlock(&owned_lock);
  return false;
}

void *allocate(size_t size, New original) {
  if (!original) fail("[sc2-icu-compat] original ICU allocation target unavailable\n");
  if (regex_depth) {
    void *p = malloc(size ? size : 1);
    if (!p) fail("[sc2-icu-compat] allocation failed\n");
    remember(p);
    return p;
  }
  return original(size);
}

void release(void *p, Delete original) {
  if (!p) return;
  if (forget(p)) { free(p); return; }
  if (!original) fail("[sc2-icu-compat] original delete unavailable\n");
  original(p);
}

void release_sized(void *p, size_t size, SizedDelete sized, Delete ordinary) {
  if (!p) return;
  if (forget(p)) { free(p); return; }
  if (sized) { sized(p, size); return; }
  if (!ordinary) fail("[sc2-icu-compat] original sized delete unavailable\n");
  ordinary(p);
}

// The shared-cache weak C++ bindings are not updated by static interposition on
// this OS. Update matching import pointers in this process's private data pages.
// Executable pages and on-disk binaries are never written.
bool patch_pointer(uintptr_t address, const InterposePair *pairs, size_t count) {
  mach_vm_address_t region = address;
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  kern_return_t kr = mach_vm_region(mach_task_self(), &region, &region_size,
      VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &info_count, &object);
  if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
  if (kr != KERN_SUCCESS || address < region || address + sizeof(void *) > region + region_size ||
      !(info.protection & VM_PROT_READ) || (info.protection & VM_PROT_EXECUTE)) return false;
  void **slot = reinterpret_cast<void **>(address);
  void *before = *slot;
  const void *replacement = nullptr;
  for (size_t n = 0; n < count; ++n)
    if (pairs[n].replacee && before == pairs[n].replacee) replacement = pairs[n].replacement;
  if (!replacement) return false;
  const mach_vm_address_t page = address & ~(mach_vm_address_t(vm_page_size) - 1);
  kr = mach_vm_protect(mach_task_self(), page, vm_page_size, false,
      VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY);
  if (kr != KERN_SUCCESS) fail("[sc2-icu-compat] import pointer copy-on-write failed\n");
  __atomic_store_n(slot, const_cast<void *>(replacement), __ATOMIC_RELEASE);
  kr = mach_vm_protect(mach_task_self(), page, vm_page_size, false, info.protection);
  if (kr != KERN_SUCCESS) fail("[sc2-icu-compat] could not restore data-page protection\n");
  return true;
}

unsigned patch_call_imports(const void *function, const InterposePair *pairs, size_t count) {
  if (!function) return 0;
  auto *code = static_cast<const unsigned char *>(function);
  unsigned changes = 0;
  for (unsigned i = 0; i < 256; ++i) {
    if (code[i] == 0xff && (code[i + 1] == 0x15 || code[i + 1] == 0x25)) {
      int32_t displacement;
      memcpy(&displacement, code + i + 2, sizeof(displacement));
      uintptr_t slot = reinterpret_cast<uintptr_t>(code + i + 6) + intptr_t(displacement);
      changes += patch_pointer(slot, pairs, count);
    }
  }
  return changes;
}

unsigned patch_data_imports(const InterposePair *pairs, size_t count) {
  auto *header = reinterpret_cast<const mach_header_64 *>(icu_header);
  const unsigned char *cmd = reinterpret_cast<const unsigned char *>(header + 1);
  intptr_t slide = 0;
  for (uint32_t i = 0; i < header->ncmds; ++i) {
    auto *lc = reinterpret_cast<const load_command *>(cmd);
    if (lc->cmd == LC_SEGMENT_64) {
      auto *seg = reinterpret_cast<const segment_command_64 *>(cmd);
      if (!strncmp(seg->segname, "__TEXT", 16))
        slide = reinterpret_cast<intptr_t>(header) - intptr_t(seg->vmaddr);
    }
    cmd += lc->cmdsize;
  }
  cmd = reinterpret_cast<const unsigned char *>(header + 1);
  unsigned changes = 0;
  for (uint32_t i = 0; i < header->ncmds; ++i) {
    auto *lc = reinterpret_cast<const load_command *>(cmd);
    if (lc->cmd == LC_SEGMENT_64) {
      auto *seg = reinterpret_cast<const segment_command_64 *>(cmd);
      if (!strncmp(seg->segname, "__DATA", 6)) {
        uintptr_t begin = uintptr_t(seg->vmaddr + slide);
        uintptr_t end = begin + seg->vmsize;
        for (uintptr_t slot = begin; slot + sizeof(void *) <= end; slot += sizeof(void *)) {
          void *value = *reinterpret_cast<void **>(slot);
          for (size_t n = 0; n < count; ++n)
            if (pairs[n].replacee && value == pairs[n].replacee) {
              changes += patch_pointer(slot, pairs, count);
              break;
            }
        }
      }
    }
    cmd += lc->cmdsize;
  }
  return changes;
}
} // namespace

extern "C" void *compat_new(size_t n) { return allocate(n, original_new); }
extern "C" void *compat_new_array(size_t n) { return allocate(n, original_new_array); }
extern "C" void compat_delete(void *p) { release(p, original_delete); }
extern "C" void compat_delete_array(void *p) { release(p, original_delete_array); }
extern "C" void compat_sized_delete(void *p, size_t n) { release_sized(p, n, original_sized_delete, original_delete); }
extern "C" void compat_sized_delete_array(void *p, size_t n) { release_sized(p, n, original_sized_delete_array, original_delete_array); }

extern "C" void *compat_regex(const uint16_t *pattern, int32_t length,
                               uint32_t flags, void *parse_error, int32_t *status) {
  if (!original_regex) fail("[sc2-icu-compat] original ICU function unavailable\n");
  if (!icu_rebound) {
    if (!dynamic_interpose || !icu_header)
      fail("[sc2-icu-compat] image-scoped rebinding unavailable\n");
    const InterposePair pairs[] = {
      {reinterpret_cast<void *>(compat_new), reinterpret_cast<void *>(original_new)},
      {reinterpret_cast<void *>(compat_new_array), reinterpret_cast<void *>(original_new_array)},
      {reinterpret_cast<void *>(compat_delete), reinterpret_cast<void *>(original_delete)},
      {reinterpret_cast<void *>(compat_delete_array), reinterpret_cast<void *>(original_delete_array)},
      {reinterpret_cast<void *>(compat_sized_delete), reinterpret_cast<void *>(original_sized_delete)},
      {reinterpret_cast<void *>(compat_sized_delete_array), reinterpret_cast<void *>(original_sized_delete_array)},
    };
    dynamic_interpose(icu_header, pairs, sizeof(pairs) / sizeof(pairs[0]));
    unsigned changes = patch_data_imports(pairs, sizeof(pairs) / sizeof(pairs[0]));
    changes += patch_call_imports(reinterpret_cast<void *>(original_regex), pairs, sizeof(pairs) / sizeof(pairs[0]));
    changes += patch_call_imports(dlsym(RTLD_NEXT, "uregex_close"), pairs, sizeof(pairs) / sizeof(pairs[0]));
    char result[128];
    int result_size = snprintf(result, sizeof(result), "[sc2-icu-compat] updated %u allocator import pointers\n", changes);
    if (result_size > 0) write(STDERR_FILENO, result, size_t(result_size));
    icu_rebound = true;
    const char notice[] = "[sc2-icu-compat] ICU image bindings updated\n";
    write(STDERR_FILENO, notice, sizeof(notice) - 1);
  }
  ++regex_depth;
  void *result = original_regex(pattern, length, flags, parse_error, status);
  --regex_depth;
  return result;
}

__attribute__((constructor)) static void initialize() {
  // dyld excludes the interposing image's own symbol references from its
  // replacements. dlsym(), by contrast, returns the interposed address.
  original_new = static_cast<New>(&::operator new);
  original_new_array = static_cast<New>(&::operator new[]);
  original_delete = static_cast<Delete>(&::operator delete);
  original_delete_array = static_cast<Delete>(&::operator delete[]);
  original_sized_delete = static_cast<SizedDelete>(&::operator delete);
  original_sized_delete_array = static_cast<SizedDelete>(&::operator delete[]);
  original_regex = &uregex_open;
  dynamic_interpose = reinterpret_cast<DynamicInterpose>(dlsym(RTLD_DEFAULT, "dyld_dynamic_interpose"));
  Dl_info icu_info{};
  if (dladdr(reinterpret_cast<void *>(original_regex), &icu_info))
    icu_header = static_cast<const mach_header *>(icu_info.dli_fbase);
  char diagnostic[512];
  int diagnostic_size = snprintf(diagnostic, sizeof(diagnostic),
      "[sc2-icu-compat] new=%p/%p new_array=%p/%p delete=%p/%p delete_array=%p/%p regex=%p/%p\n",
      reinterpret_cast<void *>(original_new), reinterpret_cast<void *>(compat_new),
      reinterpret_cast<void *>(original_new_array), reinterpret_cast<void *>(compat_new_array),
      reinterpret_cast<void *>(original_delete), reinterpret_cast<void *>(compat_delete),
      reinterpret_cast<void *>(original_delete_array), reinterpret_cast<void *>(compat_delete_array),
      reinterpret_cast<void *>(original_regex), reinterpret_cast<void *>(compat_regex));
  if (diagnostic_size > 0) write(STDERR_FILENO, diagnostic, size_t(diagnostic_size));
  if (!original_new || !original_new_array || !original_delete || !original_delete_array ||
      original_new == compat_new || original_delete == compat_delete ||
      !original_regex || original_regex == compat_regex)
    fail("[sc2-icu-compat] cannot resolve original functions; stopping experiment\n");
  const char message[] = "[sc2-icu-compat] v2 ICU-only allocation bridge loaded\n";
  write(STDERR_FILENO, message, sizeof(message) - 1);
  // dyld has already loaded this library. Do not pass the injection setting to
  // helpers that SC2 may start later (crash reporter, system_profiler, etc.).
  unsetenv("DYLD_INSERT_LIBRARIES");
}

#define INTERPOSE(replacement, target, name) \
  __attribute__((used, section("__DATA,__interpose"))) \
  static const struct { const void *replacement; const void *replacee; } name = \
    { reinterpret_cast<const void *>(replacement), reinterpret_cast<const void *>(target) }

INTERPOSE(compat_regex, &uregex_open, regex_interpose);

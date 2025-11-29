# نظام السقالات Mishkah Scaffold System

## 🎯 الهدف

نظام ذكي لتحميل مكتبات Mishkah بشكل مشروط حسب الحاجة، مع الحفاظ على صغر حجم المكتبة الأساسية.

## ⚡ التثبيت السريع

### الطريقة 1: Auto-Boot (التلقائي)
```html
<!-- إعداد الإعدادات قبل التحميل -->
<script>
  window.__MISHKAH_CONFIG__ = {
    mode: 'dev'  // dev | prod | debug | minimal
  };
</script>

<!-- تحميل نظام السقالات -->
<script src="/static/lib/mishkah.scaffold.js"></script>

<!-- ✅ الآن جميع المكتبات محمّلة تلقائياً حسب الوضع -->
```

### الطريقة 2: Manual Boot (يدوي)
```html
<script>
  window.__MISHKAH_MANUAL_BOOT__ = true;
</script>
<script src="/static/lib/mishkah.scaffold.js"></script>

<script>
  // التحكم اليدوي بالتحميل
  MishkahScaffold.boot({
    mode: 'prod',
    diagnostics: { div: false, help: false }
  }, function(err, result) {
    if (err) {
      console.error('فشل التحميل:', err);
    } else {
      console.log('تم التحميل بنجاح:', result);
    }
  });
</script>
```

## 🔧 الأوضاع المتاحة

### Development Mode (dev)
```javascript
window.__MISHKAH_CONFIG__ = { mode: 'dev' };
```
**يحمّل:**
- ✅ Core, Utils, UI, HTMLx
- ✅ Div (RuleCenter)
- ✅ Help System

### Production Mode (prod)
```javascript
window.__MISHKAH_CONFIG__ = { mode: 'prod' };
```
**يحمّل:**
- ✅ Core, Utils, UI, HTMLx
- ❌ Div (RuleCenter)
- ❌ Help System
- ❌ Performance Monitor

### Debug Mode (debug)
```javascript
window.__MISHKAH_CONFIG__ = { mode: 'debug' };
```
**يحمّل:**
- ✅ Core, Utils, UI, HTMLx
- ✅ Div (RuleCenter)
- ✅ Help System
- ✅ Performance Monitor
- ✅ Security Checks

### Minimal Mode (minimal)
```javascript
window.__MISHKAH_CONFIG__ = { mode: 'minimal' };
```
**يحمّل:**
- ✅ Core, Utils, UI, HTMLx فقط
- ❌ جميع الطبقات التشخيصية

## ⚙️ إعدادات مخصصة

### تحديد المكتبات بدقة
```javascript
window.__MISHKAH_CONFIG__ = {
  mode: 'custom',
  features: {
    core: true,
    utils: true,
    ui: true,
    htmlx: true,
    store: true,      // ✅ تفعيل
    crud: false,      // ❌ تعطيل
    pages: false
  },
  diagnostics: {
    div: true,        // ✅ تفعيل قواعد RuleCenter
    help: true,       // ✅ تفعيل نظام المساعدة
    performance: false,
    security: true
  }
};
```

### Callbacks والتحكم بالتحميل
```javascript
window.__MISHKAH_CONFIG__ = {
  mode: 'dev',
  
  // عند اكتمال التحميل
  onReady: function(info) {
    console.log('✅ تم تحميل', info.loaded.length, 'مكتبات');
    console.log('❌ فشل تحميل', info.failed.length, 'مكتبات');
    
    // ابدأ تطبيقك هنا
    Mishkah.init();
  },
  
  // عند حدوث خطأ
  onError: function(error, item) {
    console.error('خطأ في تحميل:', item.name, error);
  },
  
  // أثناء التحميل
  onProgress: function(info) {
    console.log('التقدم:', info.current, '/', info.total);
  }
};
```

### استخدام CDN
```javascript
window.__MISHKAH_CONFIG__ = {
  mode: 'prod',
  cdn: {
    enabled: true,
    baseUrl: 'https://cdn.example.com/mishkah/v1.0/'
  }
};
```

## 🌐 استخدام URL Parameters

مفيد للتطوير والتشخيص السريع:

```
https://yourapp.com/?mishkah_mode=debug
https://yourapp.com/?mishkah_debug=true
```

## 📊 التحقق من حالة التحميل

### في Console
```javascript
// عرض الإعدادات الحالية
M.help.config();

// عرض حالة التحميل
M.help.scaffold();

// الحصول على الإعدادات برمجياً
var config = MishkahScaffold.getConfig();
console.log(config);

// الحصول على حالة التحميل
var status = MishkahScaffold.getStatus();
console.log(status.loaded);  // المكتبات المحمّلة
console.log(status.failed);  // المكتبات الفاشلة
```

### الاستماع لحدث التحميل
```javascript
window.addEventListener('mishkah:ready', function(event) {
  console.log('✅ Mishkah جاهز!');
  console.log('المحمّل:', event.detail.loaded);
  console.log('الفاشل:', event.detail.failed);
});
```

## 🎨 أمثلة عملية

### مثال 1: تطبيق بسيط (Production)
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>تطبيق Mishkah</title>
  
  <script>
    window.__MISHKAH_CONFIG__ = {
      mode: 'prod',
      diagnostics: { div: false, help: false }
    };
  </script>
  <script src="/static/lib/mishkah.scaffold.js"></script>
</head>
<body>
  <div id="app"></div>
  
  <script>
    window.addEventListener('mishkah:ready', function() {
      // تطبيقك هنا
      var app = M.h('div', { class: 'container' }, [
        M.h('h1', {}, 'مرحباً بك!')
      ]);
      M.render(app, document.getElementById('app'));
    });
  </script>
</body>
</html>
```

### مثال 2: التطوير مع التشخيص الكامل
```html
<script>
  window.__MISHKAH_CONFIG__ = {
    mode: 'debug',
    onReady: function(info) {
      console.log('🎉 Mishkah جاهز');
      
      // عرض معلومات التحميل
      M.help.scaffold();
      
      // بدء التطبيق
      initApp();
    }
  };
</script>
<script src="/static/lib/mishkah.scaffold.js"></script>
```

### مثال 3: تحميل انتقائي
```html
<script>
  window.__MISHKAH_CONFIG__ = {
    mode: 'custom',
    features: {
      core: true,
      utils: true,
      ui: false,      // ❌ لا نحتاج UI
      htmlx: true,
      store: true,
      crud: true,
      pages: false
    },
    diagnostics: {
      div: true,
      help: false
    }
  };
</script>
<script src="/static/lib/mishkah.scaffold.js"></script>
```

## 🔄 إعادة التحميل

```javascript
// إعادة تحميل بإعدادات جديدة
MishkahScaffold.reload({
  mode: 'prod'
}, function(err, result) {
  if (!err) {
    console.log('✅ تمت إعادة التحميل');
  }
});
```

## 📝 الترتيب الصحيح للتحميل

نظام السقالات يضمن تحميل المكتبات بالترتيب الصحيح:

1. **Core** - القلب الأساسي
2. **Utils** - الأدوات المساعدة
3. **UI** - المكونات
4. **HTMLx** - التحليل والعرض
5. **Store** - إدارة الحالة
6. **CRUD** - عمليات قاعدة البيانات
7. **Pages** - المسارات والصفحات
8. **Div** - نظام القواعد (تشخيصي)
9. **Help** - نظام المساعدة (تشخيصي)
10. **Performance** - مراقبة الأداء (تشخيصي)

## 💡 نصائح

### للإنتاج (Production)
- استخدم `mode: 'prod'` لتعطيل الطبقات التشخيصية
- فعّل CDN إذا كان متاحاً
- عطّل `diagnostics` كلياً

### للتطوير (Development)
- استخدم `mode: 'dev'` للحصول على المساعدة
- فعّل `diagnostics.div` للتحقق من القواعد
- فعّل `diagnostics.help` للمساعدة السريعة

### للتشخيص (Debugging)
- استخدم `mode: 'debug'` لتفعيل كل شيء
- استخدم URL parameters للتبديل السريع
- راقب console للأخطاء

## 🚀 الميزات الرئيسية

✅ **تحميل مشروط** - فقط ما تحتاجه
✅ **أوضاع معرّفة مسبقاً** - dev, prod, debug, minimal
✅ **إعدادات مرنة** - تحكم كامل بكل مكتبة
✅ **Callbacks** - onReady, onError, onProgress
✅ **إعادة المحاولة** - retry logic للتحميل
✅ **CDN Support** - دعم شبكات CDN
✅ **URL Parameters** - للتشخيص السريع
✅ **Events** - mishkah:ready event
✅ **صغير** - أقل من 10KB

## 🔒 الأمان

- تحميل من نفس الأصل (Same Origin) افتراضياً
- دعم CSP (Content Security Policy)
- Timeout للتحميل (10 ثواني افتراضياً)
- إعادة المحاولة المحدودة (مرتين افتراضياً)

---

**للمزيد من المعلومات:**
- `M.help()` - التعليمات الرئيسية
- `M.help.config()` - الإعدادات الحالية
- `M.help.scaffold()` - حالة التحميل

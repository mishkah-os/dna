# HTMLx - Template-Based Development

## 📖 Introduction

HTMLx is Mishkah's **template-based** approach to building UIs. It combines the familiarity of HTML with powerful state binding and component resolution.

---

## 📚 Documentation Files

### Core Guides
- **[htmlx-guide.md](./htmlx-guide.md)** - Complete writing guide with examples
- **[mishkah-htmlx-js.md](./mishkah-htmlx-js.md)** - Technical deep-dive (149KB file)

---

## 🚀 Quick Start

```html
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <script>
  window.MishkahAutoConfig = { css: 'mi' };
  </script>
  <script src="/lib/mishkah.js" data-htmlx data-css="mi"></script>
</head>
<body>
  <div id="app"></div>
  
  <template id="main">
    <script type="application/json" data-m-data data-m-path="data">
      { "message": "Hello Mishkah!" }
    </script>
    
    <div data-m-scope="main">
      <h1>{state.data.message}</h1>
    </div>
  </template>
  
  <script>
  MishkahAuto.ready(M => {
    M.app.make({}, {
      templateId: 'main',
      mount: '#app'
    });
  });
  </script>
</body>
</html>
```

---

## ✨ Key Features

### 1. State Interpolation
```html
<h1>{state.data.title}</h1>
<p>{state.env.theme}</p>
```

### 2. Directives
```html
<div x-if="state.data.isLoggedIn">Welcome!</div>
<li x-for="user in state.data.users">{user.name}</li>
```

### 3. Component Resolution
```html
<!-- All 4 styles work -->
<Modal title="Hello" />
<modal title="Hello" />
<m-modal title="Hello" />
<comp-Modal title="Hello" />
```

### 4. Data Scripts
```html
<script type="application/json" data-m-data data-m-path="data">
  { "count": 0 }
</script>
```

---

## 🎯 When to Use HTMLx

**✅ Best for:**
- Rapid prototyping
- Content-heavy applications
- Designer collaboration
- Simple state management

**❌ Consider DSL instead for:**
- Complex business logic
- Type-safe development
- Large team projects
- Maximum performance

---

## 📖 Full Documentation

See [htmlx-guide.md](./htmlx-guide.md) for:
- Complete syntax reference
- AJAX integration
- Event handling
- Scoped styles
- Best practices
- Common mistakes

---

## 🚨 Common Issues

### White Screen?
1. Check `data-htmlx` attribute exists
2. Validate JSON in `<script>` tags
3. Ensure `data-m-path` is present
4. Verify template ID matches

### Not Updating?
1. Check console for errors
2. Verify `setState` returns new state
3. Ensure template is properly mounted

---

**Next:** [Writing Guide](./htmlx-guide.md) | [Back to Main](../README.md)

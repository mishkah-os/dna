# mishkah-htmlx.js - HTMLx Template Engine

## 📄 File Info
- **Size**: 149KB (4162 lines)
- **Role**: Template parser, component resolver, state binding
- **Critical**: **Essential** for rendering - without it → white screen

---

## 🎯 Core Responsibilities

### 1. **Template Parsing**
Converts `<template>` tags into DOM with state bindings.

### 2. **Component Resolution**
Supports multiple naming conventions:
- `<Modal>` → JSX-style (PascalCase)
- `<modal>` → Vue-style (lowercase)
- `<m-modal>` → Web Component (prefix)
- `<comp-Modal>` → Legacy

### 3. **State Binding**
- `{state.data.count}` interpolation
- `x-if`, `x-for`, `x-bind` directives
- JSON script tags for data/env

### 4. **CSS Scoping**
- Scopes styles to namespace
- `:host` selector support

---

## 📝 Template Anatomy

### Basic Template Structure

```html
<template id="my-app">
  <!-- 1. Environment (optional) -->
  <script type="application/json" data-m-env>
    {
      "theme": "dark",
      "lang": "ar",
      "dir": "rtl"
    }
  </script>
  
  <!-- 2. Data (state) -->
  <script type="application/json" data-m-data data-m-path="data">
    {
      "count": 0,
      "users": []
    }
  </script>
  
  <!-- 3. AJAX Configuration (optional) -->
  <script type="application/json" 
          data-m-ajax-map
          data-for="loadUsers">
    {
      "loadUsers": {
        "url": "/api/users",
        "method": "GET"
      }
    }
  </script>
  
  <!--  4. Scoped Styles (optional) -->
  <style>
    :host {
      display: block;
      padding: 1rem;
    }
    .counter {
      font-size: 2rem;
    }
  </style>
  
  <!-- 5. HTML Content -->
  <div data-m-scope="my-app">
    <h1 class="counter">Count: {state.data.count}</h1>
    <button data-m-order="increment">+</button>
    <button data-m-order="loadUsers">Load Users</button>
  </div>
</template>
```

---

## 🔧 Template Attributes

### Core Attributes

#### `data-namespace`
Unique ID for the template (defaults to `id` if missing).

```html
<template data-namespace="app-v2" id="my-app">
  ...
</template>
```

#### `data-mount`
Auto-mount target selector.

```html
<template data-mount="#app">
  ...
</template>
```

#### `data-m-scope`
Scope ID for CSS isolation.

```html
<div data-m-scope="my-component">
  ...
</div>
```

---

## 📊 Data Scripts

### `data-m-env` - Environment Config

```html
<script type="application/json" data-m-env>
  {
    "theme": "dark",
    "lang": "ar",
    "dir": "rtl",
    "css": "mi",
    "title": "My App"
  }
</script>
```

**Merges into:** `state.env`

---

### `data-m-data` - Application Data

**Required Attribute:** `data-m-path`

```html
<script type="application/json" 
        data-m-data 
        data-m-path="data">
  {
    "count": 0,
    "users": []
  }
</script>
```

**Path Examples:**
- `data` → `state.data = {...}`
- `data.users` → `state.data.users = [...]`
- `data.settings.theme` → `state.data.settings.theme = "..."`

---

### `data-m-ajax` - AJAX Integration

#### Inline Configuration
```html
<script type="application/json"
        data-m-data
        data-m-path="data.users"
        data-m-ajax='{"url": "/api/users", "method": "GET"}'>
  []
</script>
```

#### Map Configuration
```html
<script type="application/json"
        data-m-ajax-map
        data-for="loadUsers">
  {
    "loadUsers": {
      "url": "/api/users",
      "method": "GET",
      "responsePath": "data.users",
      "auto": false
    }
  }
</script>
```

**AJAX Options:**
- `url`: Endpoint URL
- `method`: HTTP method (`GET`, `POST`, etc.)
- `responsePath`: Path to extract from response
- `assign`: Where to assign response data
- `auto`: Auto-execute on load (default: `false`)
- `mode`: AJAX mode (`fetch`, `xhr`)
- `vars`: Variables to interpolate (`{{vars.userId}}`)

**Placeholder Support:**
```json
{
  "url": "/api/users/{{vars.userId}}",
  "vars": { "userId": 123 }
}
```

---

## 🔀 State Interpolation

### Syntax: `{state.path.to.value}`

```html
<h1>{state.env.title}</h1>
<p>Count: {state.data.count}</p>
<span>{state.i18n.greeting}</span>
```

### Supported Paths
- `state.env.*` - Environment
- `state.data.*` - Application data
- `state.i18n.*` - Translations
- `state.head.*` - Head metadata

---

## 🎨 Directives (HTMLx)

### `x-if` - Conditional Rendering
```html
<div x-if="state.data.isLoggedIn">
  Welcome back!
</div>

<div x-if="state.data.count > 0">
  Count is positive
</div>
```

**Evaluated as JavaScript expression with `state` in scope.**

---

### `x-for` - Loop Rendering
```html
<div x-for="user in state.data.users" key="user.id">
  <p>{user.name}</p>
</div>
```

**Note:** Mishkah's `x-for` has limitations - not fully reactive like Vue/Alpine.

---

### `x-bind` - Dynamic Attributes
```html
<img x-bind:src="state.data.imageUrl" />
<input x-bind:value="state.data.username" />
```

---

### `x-class` - Dynamic Classes
```html
<div x-class="state.data.isActive ? 'active' : 'inactive'">
  ...
</div>
```

---

## 🧩 Component Resolution

### Naming Conventions

HTMLx supports **4 naming styles**:

#### 1. JSX-style (PascalCase)
```html
<Modal title="Hello">
  <p>Content</p>
</Modal>
```
→ Looks for `Mishkah.UI.Modal`

#### 2. Vue-style (lowercase)
```html
<modal title="Hello">
  <p>Content</p>
</modal>
```
→ Converts to `Modal`, looks for `Mishkah.UI.Modal`

#### 3. Web Component (prefix `m-`)
```html
<m-modal title="Hello">
  <p>Content</p>
</m-modal>
```
→ Strips `m-`, converts to `Modal`

#### 4. Legacy (prefix `comp-`)
```html
<comp-Modal title="Hello">
  <p>Content</p>
</comp-Modal>
```
→ Strips `comp-`, looks for `Modal`

---

## 🚨 Debugging White Screens

### Why HTMLx Causes White Screens

HTMLx **fails silently** by design. Common causes:

#### 1. **Missing Template**
**Symptom:** Blank screen, no errors.

**Check:**
```javascript
console.log('Template exists:', !!document.getElementById('my-app'));
```

**Fix:** Ensure `<template id="my-app">` exists in HTML.

---

#### 2. **JSON Parse Errors**
**Symptom:** Console warning: `data-m-data ليس JSON صالحًا`.

**Common Mistakes:**
```html
<!-- ❌ Trailing comma -->
<script type="application/json" data-m-data data-m-path="data">
  { "count": 0, }
</script>

<!-- ❌ Single quotes -->
<script type="application/json" data-m-data data-m-path="data">
  { 'count': 0 }
</script>

<!-- ✅ Correct -->
<script type="application/json" data-m-data data-m-path="data">
  { "count": 0 }
</script>
```

---

#### 3. **Missing `data-m-path`**
```html
<!-- ❌ Missing path -->
<script type="application/json" data-m-data>
  { "count": 0 }
</script>

<!-- ✅ Correct -->
<script type="application/json" data-m-data data-m-path="data">
  { "count": 0 }
</script>
```

---

#### 4. **Component Not Found**
**Symptom:** Element renders as plain HTML, not component.

**Check:**
```javascript
console.log('Modal exists:', typeof Mishkah.UI.Modal);
```

**Fix:** Register component:
```javascript
Mishkah.UI.Modal = function(props) { ... };
```

---

#### 5. **CSS Scoping Issues**
**Symptom:** Styles not applied.

**Check:** Ensure `data-m-scope` matches namespace:
```html
<template id="my-app">
  <div data-m-scope="my-app">
    ...
  </div>
</template>
```

---

## 🔍 Diagnostic Logging

### Enable HTMLx Debugging

```javascript
// Before loading mishkah.js
window.MishkahAutoConfig = {
  devtools: true,  // Enables dev panel
  debug: true       // Verbose logging
};
```

### Manual Inspection
```javascript
// Check if HTMLx loaded
console.log('HTMLx:', window.Mishkah?.HTMLx);
console.log('HTMLxAgent:', window.Mishkah?.HTMLxAgent);

// Inspect template parsing
const template = document.getElementById('my-app');
const parts = extractTemplateParts(template);
console.log('Parsed:', parts);
```

---

## 📚 Related Files
- [mishkah.js](./mishkah-js.md) - Auto-loader (loads HTMLx)
- [mishkah.core.js](./mishkah-core-js.md) - App engine (uses HTMLx)
- [htmlx-guide.md](./htmlx-guide.md) - Writing HTMLx templates

---

## ✅ Best Practices

### 1. **Always Use `data-m-path`**
```html
<script type="application/json" data-m-data data-m-path="data">
  { ... }
</script>
```

### 2. **Validate JSON**
Use a JSON validator before pasting into `<script>` tags.

### 3. **Use Namespace**
```html
<template data-namespace="app-v1" id="my-app">
  ...
</template>
```

### 4. **Scope CSS**
```html
<div data-m-scope="my-app">
  ...
</div>
```

### 5. **Console Log State**
```javascript
M.app.make({}, {
  templateId: 'my-app',
  mount: '#app'
}).then(app => {
  console.log('Initial state:', app.getState());
});
```

---

## 🏁 Summary

HTMLx is the **core template engine** that:
- Parses `<template>` tags
- Binds state to DOM
- Resolves components
- Scopes CSS

**Without HTMLx → White Screen.**

**Key Requirements:**
1. `data-htmlx` on `<script src="mishkah.js">`
2. Valid JSON in `<script>` tags
3. Correct `data-m-path` attributes
4. Template ID matches `templateId` option

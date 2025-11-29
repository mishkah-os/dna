# HTMLx Writing Guide - المرجع الكامل

## 📝 Introduction

HTMLx هو نظام templates يجمع بين:
- **JSX** - Component-based architecture
- **Vue** - Declarative directives
- **Web Components** - Standard browser APIs

---

## 🎯 Basic Template

### Minimal Template

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
      { "message": "مرحباً" }
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

## 📊 State Structure

### Full State Schema

```json
{
  "env": {
    "theme": "dark",
    "lang": "ar",
    "dir": "rtl",
    "css": "mi",
    "cssLibrary": "mi",
    "title": "App Title"
  },
  "data": {
    "users": [],
    "count": 0,
    "settings": {}
  },
  "i18n": {
    "lang": "ar",
    "dict": {
      "greeting": {
        "ar": "مرحباً",
        "en": "Hello"
      }
    }
  },
  "head": {
    "title": "Page Title",
    "meta": {}
  }
}
```

---

## 🔧 Data Scripts

### 1. Environment (`data-m-env`)

```html
<script type="application/json" data-m-env>
  {
    "theme": "dark",
    "lang": "ar",
    "dir": "rtl"
  }
</script>
```

**Merges into:** `state.env`

---

### 2. Data (`data-m-data`)

**Single Path:**
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

**Nested Path:**
```html
<script type="application/json" 
        data-m-data 
        data-m-path="data.settings">
  {
    "darkMode": true,
    "fontSize": 16
  }
</script>
```

---

### 3. i18n (`data-m-i18n`)

```html
<script type="application/json" data-m-i18n>
  {
    "lang": "ar",
    "dict": {
      "welcome": {
        "ar": "مرحباً",
        "en": "Welcome"
      },
      "goodbye": {
        "ar": "وداعاً",
        "en": "Goodbye"
      }
    }
  }
</script>
```

**Access:** `{state.i18n.dict.welcome[state.i18n.lang]}`

---

## 🌐 AJAX Integration

### Inline AJAX

```html
<script type="application/json"
        data-m-data
        data-m-path="data.users"
        data-m-ajax='{"url": "/api/users", "method": "GET"}'>
  []
</script>
```

### AJAX Map (Recommended)

```html
<!-- Define AJAX configs -->
<script type="application/json"
        data-m-ajax-map>
  {
    "loadUsers": {
      "url": "/api/users",
      "method": "GET",
      "assign": "data.users",
      "auto": false
    },
    "loadPosts": {
      "url": "/api/posts",
      "method": "GET",
      "assign": "data.posts"
    }
  }
</script>

<!-- Empty data, loaded via AJAX -->
<script type="application/json" 
        data-m-data 
        data-m-path="data.users">
  []
</script>
```

### Execute AJAX

```javascript
M.app.make({}, {
  templateId: 'main',
  mount: '#app',
  ajax: {
    loadUsers: { auto: true }  // Auto-execute on mount
  }
});
```

Or programmatically:
```javascript
app.getState().ajax.loadUsers().then(data => {
  console.log('Loaded:', data);
});
```

---

## 🎨 Interpolation

### Basic Syntax

```html
<h1>{state.data.title}</h1>
<p>{state.env.theme}</p>
<span>{state.data.count}</span>
```

### Nested Access

```html
<div>{state.data.user.name}</div>
<div>{state.data.settings.darkMode}</div>
```

### Array Access

```html
<div>{state.data.users[0].name}</div>
```

---

## 🔀 Directives

### `x-if` - Conditional

```html
<div x-if="state.data.isLoggedIn">
  <p>Welcome back!</p>
</div>

<div x-if="state.data.count > 10">
  <p>Count is high</p>
</div>

<div x-if="state.env.theme === 'dark'">
  <p>Dark mode active</p>
</div>
```

---

### `x-for` - Loop

```html
<ul>
  <li x-for="user in state.data.users" key="user.id">
    {user.name} - {user.email}
  </li>
</ul>
```

**Note:** `key` attribute recommended for performance.

---

### `x-bind` - Dynamic Attributes

```html
<img x-bind:src="state.data.image Url" />
<input x-bind:value="state.data.username" />
<a x-bind:href="state.data.link">Click</a>
```

---

### `x-class` - Dynamic Classes

```html
<div x-class="state.data.isActive ? 'active' : 'inactive'">
  ...
</div>

<button x-class="state.data.count > 0 ? 'btn-primary' : 'btn-secondary'">
  Click
</button>
```

---

## 🧩 Components

### Registering Components

```javascript
// In Mishkah.UI namespace
Mishkah.UI.Modal = function(props, children) {
  return h('div', { class: 'modal' }, [
    h('div', { class: 'modal-header' }, props.title),
    h('div', { class: 'modal-body' }, children)
  ]);
};
```

### Using Components

```html
<!-- JSX-style -->
<Modal title="Hello">
  <p>Content here</p>
</Modal>

<!-- Vue-style -->
<modal title="Hello">
  <p>Content here</p>
</modal>

<!-- Web Component -->
<m-modal title="Hello">
  <p>Content here</p>
</m-modal>
```

---

## 🎯 Event Handling

### Using `data-m-order`

```html
<button data-m-order="increment">+</button>
<button data-m-order="decrement">-</button>
<button data-m-order="reset">Reset</button>
```

### Defining Orders

```javascript
M.app.make({}, {
  templateId: 'main',
  mount: '#app',
  orders: {
    increment: {
      on: ['click'],
      handler: (e, app) => {
        app.setState(s => {
          s.data.count++;
          return s;
        });
      }
    },
    decrement: {
      on: ['click'],
      handler: (e, app) => {
        app.setState(s => {
          s.data.count--;
          return s;
        });
      }
    },
    reset: {
      on: ['click'],
      handler: (e, app) => {
        app.setState(s => {
          s.data.count = 0;
          return s;
        });
      }
    }
  }
});
```

---

## 🎨 Scoped Styles

### Using `:host` Selector

```html
<template id="my-component">
  <style>
    :host {
      display: block;
      padding: 1rem;
      background: var(--surface);
    }
    
    .title {
      font-size: 2rem;
      color: var(--primary);
    }
    
    .button {
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
    }
  </style>
  
  <div data-m-scope="my-component">
    <h1 class="title">{state.data.title}</h1>
    <button class="button">Click</button>
  </div>
</template>
```

**`:host` will be replaced with `[data-m-scope="my-component"]`**

---

## ✅ Best Practices

### 1. Always Use `data-m-path`
```html
<!-- ✅ Correct -->
<script type="application/json" data-m-data data-m-path="data">
  { "count": 0 }
</script>

<!-- ❌ Wrong -->
<script type="application/json" data-m-data>
  { "count": 0 }
</script>
```

### 2. Validate JSON
Use JSON validator before pasting:
```html
<!-- ❌ Trailing comma -->
{ "count": 0, }

<!-- ✅ Correct -->
{ "count": 0 }
```

### 3. Use Namespace
```html
<template data-namespace="app-v1" id="main">
  ...
</template>
```

### 4. Scope Your Styles
```html
<div data-m-scope="my-component">
  ...
</div>
```

### 5. Use Semantic Paths
```html
<!-- ✅ Good -->
<script data-m-data data-m-path="data.users">

<!-- ❌ Bad -->
<script data-m-data data-m-path="data.x">
```

---

## 🚨 Common Mistakes

### 1. Missing `data-m-path`
```html
<!-- ❌ Will show warning -->
<script type="application/json" data-m-data>
  { "count": 0 }
</script>
```

### 2. Invalid JSON
```html
<!-- ❌ Single quotes -->
<script type="application/json" data-m-data data-m-path="data">
  { 'count': 0 }
</script>

<!-- ❌ Trailing comma -->
<script type="application/json" data-m-data data-m-path="data">
  { "count": 0, }
</script>
```

### 3. Wrong Template ID
```javascript
// ❌ Mismatch
M.app.make({}, { templateId: 'main-app' });
```
```html
<template id="main">...</template>
```

### 4. Missing `data-htmlx`
```html
<!-- ❌ No HTMLx loaded -->
<script src="/lib/mishkah.js"></script>

<!-- ✅ Correct -->
<script src="/lib/mishkah.js" data-htmlx></script>
```

---

## 📚 Complete Example

```html
<!DOCTYPE html>
<html lang="ar" dir="rtl" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <title>Todo App</title>
  <script>
  window.MishkahAutoConfig = { css: 'mi' };
  </script>
  <script src="/lib/mishkah.js" data-htmlx data-css="mi"></script>
</head>
<body>
  <div id="app"></div>
  
  <template id="todo-app">
    <!-- Environment -->
    <script type="application/json" data-m-env>
      {
        "theme": "dark",
        "lang": "ar",
        "title": "Todo App"
      }
    </script>
    
    <!-- Data -->
    <script type="application/json" data-m-data data-m-path="data">
      {
        "todos": [],
        "newTodo": ""
      }
    </script>
    
    <!-- i18n -->
    <script type="application/json" data-m-i18n>
      {
        "dict": {
          "addTodo": { "ar": "إضافة", "en": "Add" },
          "placeholder": { "ar": "مهمة جديدة", "en": "New task" }
        }
      }
    </script>
    
    <!-- Styles -->
    <style>
      :host {
        display: block;
        max-width: 600px;
        margin: 2rem auto;
        padding: 2rem;
      }
      
      .todo-item {
        padding: 1rem;
        border: 1px solid var(--border);
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
      }
      
      .todo-item.completed {
        opacity: 0.6;
        text-decoration: line-through;
      }
    </style>
    
    <!-- HTML -->
    <div data-m-scope="todo-app">
      <h1>{state.env.title}</h1>
      
      <div class="input-group">
        <input 
          type="text" 
          x-bind:placeholder="state.i18n.dict.placeholder[state.env.lang]"
          data-m-order="updateNewTodo">
        <button data-m-order="addTodo">
          {state.i18n.dict.addTodo[state.env.lang]}
        </button>
      </div>
      
      <div class="todos">
        <div x-if="state.data.todos.length === 0">
          <p>لا توجد مهام</p>
        </div>
        
        <div x-if="state.data.todos.length > 0">
          <div x-for="todo in state.data.todos" key="todo.id"
               x-class="todo.completed ? 'todo-item completed' : 'todo-item'">
            <span>{todo.text}</span>
            <button data-m-order="toggleTodo" data-todo-id="{todo.id}">
              Toggle
            </button>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  MishkahAuto.ready(M => {
    M.app.make({}, {
      templateId: 'todo-app',
      mount: '#app',
      orders: {
        updateNewTodo: {
          on: ['input'],
          handler: (e, app) => {
            app.setState(s => {
              s.data.newTodo = e.target.value;
              return s;
            });
          }
        },
        addTodo: {
          on: ['click'],
          handler: (e, app) => {
            app.setState(s => {
              if (s.data.newTodo.trim()) {
                s.data.todos.push({
                  id: Date.now(),
                  text: s.data.newTodo,
                  completed: false
                });
                s.data.newTodo = '';
              }
              return s;
            });
          }
        },
        toggleTodo: {
          on: ['click'],
          handler: (e, app) => {
            const id = parseInt(e.target.dataset.todoId);
            app.setState(s => {
              const todo = s.data.todos.find(t => t.id === id);
              if (todo) todo.completed = !todo.completed;
              return s;
            });
          }
        }
      }
    });
  });
  </script>
</body>
</html>
```

---

## 🏁 Summary

HTMLx templates require:
1. ✅ `<template id="...">` tag
2. ✅ `data-m-data` with `data-m-path`
3. ✅ Valid JSON (no trailing commas, use double quotes)
4. ✅ `data-m-scope` for CSS scoping
5. ✅ `data-htmlx` attribute on `<script src="mishkah.js">`
6. ✅ Template ID matches `templateId` in `M.app.make()`

**Master these fundamentals to avoid white screens!**

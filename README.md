# 🚀 Maria Zorkaltseva - ML & AI Blog

Personal blog focused on **machine learning, AI systems, MLOps, and real-world engineering problems**.

👉 Live: https://mariazork.github.io

## ⚙️ Tech Stack

- Next.js 14 (Pages Router, Static Export)
- MDX for blog content
- Tailwind CSS
- TypeScript

## 🧪 Local Development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## 🏗 Build

```bash
npm run build
```

Generates a static site in `/out` deployed to GitHub Pages.

## 📝 Writing Posts

Blog posts are in `src/content/posts/` as MDX files with frontmatter:

```mdx
---
title: Post Title
subtitle: Optional subtitle
date: 2024-01-15
author: Maria Zorkaltseva
categories: [Machine Learning, Deep Learning]
tags: [PyTorch, Tutorial]
image: /images/post-image.jpg
published: true
---
```

### Components in MDX

```mdx
<BlogImage
  src="/images/post-image.jpg"
  alt="Description"
  caption="Figure 1: Example"
  zoomable
/>

<SectionDivider />
```
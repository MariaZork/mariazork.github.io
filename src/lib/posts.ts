import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import readingTime from 'reading-time';
import he from 'he';
import { Post, PostMeta, Category, Tag } from '@/types/post';

const postsDirectory = path.join(process.cwd(), 'src/content/posts');

export function getPostSlugs(): string[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }
  return fs.readdirSync(postsDirectory).filter(file => file.endsWith('.mdx'));
}

export function getPostBySlug(slug: string): Post {
  const realSlug = slug.replace(/\.mdx$/, '');
  const fullPath = path.join(postsDirectory, `${realSlug}.mdx`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const { data, content } = matter(fileContents);

  const readTime = readingTime(content);

  return {
    slug: realSlug,
    title: data.title ? he.decode(data.title) : '',
    subtitle: data.subtitle ? he.decode(data.subtitle) : '',
    date: data.date || new Date().toISOString(),
    author: data.author || 'Maria Zorkaltseva',
    excerpt: data.excerpt || content.slice(0, 160),
    content,
    categories: data.categories || [],
    tags: data.tags || [],
    image: data.image || '/images/default-og-image.jpg',
    published: data.published !== false,
    readingTime: readTime.text,
  };
}

export function getAllPosts(): Post[] {
  const slugs = getPostSlugs();
  const posts = slugs
    .map((slug) => getPostBySlug(slug))
    .filter((post) => post.published)
    .sort((a, b) => (new Date(b.date).getTime() - new Date(a.date).getTime()));

  return posts;
}

export function getPostsByCategory(category: string): Post[] {
  const allPosts = getAllPosts();
  return allPosts.filter(post =>
    post.categories.some(cat =>
      cat.toLowerCase().replace(/\s+/g, '-') === category.toLowerCase()
    )
  );
}

export function getPostsByTag(tag: string): Post[] {
  const allPosts = getAllPosts();
  return allPosts.filter(post =>
    post.tags.some(t =>
      t.toLowerCase().replace(/\s+/g, '-') === tag.toLowerCase()
    )
  );
}

export function getAllCategories(): Category[] {
  const allPosts = getAllPosts();
  const categoriesMap = new Map<string, number>();

  allPosts.forEach(post => {
    post.categories.forEach(category => {
      const count = categoriesMap.get(category) || 0;
      categoriesMap.set(category, count + 1);
    });
  });

  return Array.from(categoriesMap.entries()).map(([name, count]) => ({
    name,
    count,
    slug: name.toLowerCase().replace(/\s+/g, '-'),
  })).sort((a, b) => b.count - a.count);
}

export function getAllTags(): Tag[] {
  const allPosts = getAllPosts();
  const tagsMap = new Map<string, number>();

  allPosts.forEach(post => {
    post.tags.forEach(tag => {
      const count = tagsMap.get(tag) || 0;
      tagsMap.set(tag, count + 1);
    });
  });

  return Array.from(tagsMap.entries()).map(([name, count]) => ({
    name,
    count,
    slug: name.toLowerCase().replace(/\s+/g, '-'),
  })).sort((a, b) => b.count - a.count);
}

export function getRecentPosts(limit: number = 5): PostMeta[] {
  const allPosts = getAllPosts();
  return allPosts.slice(0, limit).map(post => ({
    slug: post.slug,
    title: post.title,
    subtitle: post.subtitle,
    date: post.date,
    author: post.author,
    excerpt: post.excerpt,
    categories: post.categories,
    tags: post.tags,
    image: post.image,
    readingTime: post.readingTime,
  }));
}
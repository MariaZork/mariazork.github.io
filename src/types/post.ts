export interface Post {
  slug: string;
  title: string;
  subtitle?: string;
  date: string;
  author: string;
  excerpt: string;
  content: string;
  categories: string[];
  tags: string[];
  image: string;
  published: boolean;
  readingTime?: string;
}

export interface PostMeta {
  slug: string;
  title: string;
  subtitle?: string;
  date: string;
  author: string;
  excerpt: string;
  categories: string[];
  tags: string[];
  image: string;
  readingTime?: string;
}

export interface Category {
  name: string;
  count: number;
  slug: string;
}

export interface Tag {
  name: string;
  count: number;
  slug: string;
}
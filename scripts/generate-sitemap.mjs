import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SITE_URL = 'https://mariazork.github.io';
const outDir = path.join(__dirname, '..', 'out');
const postsDir = path.join(__dirname, '..', 'src', 'content', 'posts');

function getPostsData() {
  if (!fs.existsSync(postsDir)) return [];
  
  return fs.readdirSync(postsDir)
    .filter(f => f.endsWith('.mdx'))
    .map(file => {
      const content = fs.readFileSync(path.join(postsDir, file), 'utf8');
      const dateMatch = content.match(/^date:\s*['"]?([^'">\n]+)/m);
      const publishedMatch = content.match(/^published:\s*(true|false)/m);
      
      if (publishedMatch && publishedMatch[1] === 'false') return null;
      
      const slug = file.replace(/\.mdx$/, '');
      const date = dateMatch ? dateMatch[1].trim() : new Date().toISOString();
      return { slug, date };
    })
    .filter(Boolean)
    .sort((a, b) => new Date(b.date) - new Date(a.date));
}

function generateSitemap(posts) {
  const staticPages = [
    { url: '/', priority: '1.0', changefreq: 'weekly' },
    { url: '/blog/', priority: '0.9', changefreq: 'weekly' },
    { url: '/about/', priority: '0.7', changefreq: 'monthly' },
    { url: '/projects/', priority: '0.7', changefreq: 'monthly' },
    { url: '/interview-prep/', priority: '0.7', changefreq: 'monthly' },
    { url: '/categories/', priority: '0.6', changefreq: 'weekly' },
    { url: '/tags/', priority: '0.6', changefreq: 'weekly' },
  ];

  const staticEntries = staticPages.map(p => `  <url>
    <loc>${SITE_URL}${p.url}</loc>
    <changefreq>${p.changefreq}</changefreq>
    <priority>${p.priority}</priority>
  </url>`).join('\n');

  const postEntries = posts.map(({ slug, date }) => {
    const lastmod = new Date(date).toISOString().split('T')[0];
    return `  <url>
    <loc>${SITE_URL}/blog/${slug}/</loc>
    <lastmod>${lastmod}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>`;
  }).join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${staticEntries}
${postEntries}
</urlset>`;
}

function generateRobotsTxt() {
  return `User-agent: *
Allow: /

Sitemap: ${SITE_URL}/sitemap.xml
`;
}

// Генерация
const posts = getPostsData();
const sitemap = generateSitemap(posts);
const robots = generateRobotsTxt();

fs.writeFileSync(path.join(outDir, 'sitemap.xml'), sitemap, 'utf8');
console.log(`✅ sitemap.xml — ${posts.length} posts`);

fs.writeFileSync(path.join(outDir, 'robots.txt'), robots, 'utf8');
console.log('✅ robots.txt');
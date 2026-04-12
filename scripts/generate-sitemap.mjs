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
      const tags = parseYamlList(content, 'tags');
      const categories = parseYamlList(content, 'categories');

      return { slug, date, tags, categories };
    })
    .filter(Boolean)
    .sort((a, b) => new Date(b.date) - new Date(a.date));
}

function parseYamlList(content, key) {
  const result = new Set();
  const lines = content.split('\n');
  let inBlock = false;

  // inline array: tags: [foo, bar]
  const inlineMatch = content.match(new RegExp(`^${key}:\\s*\\[([^\\]]+)\\]`, 'm'));
  if (inlineMatch) {
    inlineMatch[1].split(',').map(t => t.trim().replace(/['"]/g, '')).forEach(t => t && result.add(t));
    return Array.from(result);
  }

  // block list:
  // tags:
  //   - foo
  //   - bar
  for (const line of lines) {
    if (new RegExp(`^${key}:\\s*$`).test(line)) {
      inBlock = true;
      continue;
    }
    if (inBlock) {
      if (/^\s+-\s+/.test(line)) {
        const val = line.replace(/^\s+-\s+/, '').trim().replace(/['"]/g, '');
        if (val) result.add(val);
      } else if (line.trim() !== '' && !/^\s/.test(line)) {
        break;
      }
    }
  }

  return Array.from(result);
}

function escapeXml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function generateSitemapIndex() {
  const today = new Date().toISOString().split('T')[0];
  return `<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>${SITE_URL}/sitemap-0.xml</loc>
    <lastmod>${today}</lastmod>
  </sitemap>
</sitemapindex>`;
}

function generateSitemap0(posts, allTags, allCategories) {
  const staticPages = [
    { url: '/',              priority: '1.0', changefreq: 'weekly'  },
    { url: '/blog/',         priority: '0.9', changefreq: 'weekly'  },
    { url: '/about/',        priority: '0.8', changefreq: 'monthly' },
    { url: '/projects/',     priority: '0.7', changefreq: 'monthly' },
    { url: '/interview-prep/', priority: '0.7', changefreq: 'monthly' },
    { url: '/categories/',   priority: '0.6', changefreq: 'weekly'  },
    { url: '/tags/',         priority: '0.6', changefreq: 'weekly'  },
  ];

  const today = new Date().toISOString();

  const urls = [];

  staticPages.forEach(p => {
    urls.push(`  <url>
    <loc>${SITE_URL}${escapeXml(p.url)}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>${p.changefreq}</changefreq>
    <priority>${p.priority}</priority>
  </url>`);
  });

  allTags.forEach(tag => {
    const slug = escapeXml(tag.toLowerCase().replace(/\s+/g, '-'));
    urls.push(`  <url>
    <loc>${SITE_URL}/tags/${slug}/</loc>
    <lastmod>${today}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.6</priority>
  </url>`);
  });

  allCategories.forEach(cat => {
    const slug = escapeXml(cat.toLowerCase().replace(/\s+/g, '-'));
    urls.push(`  <url>
    <loc>${SITE_URL}/categories/${slug}/</loc>
    <lastmod>${today}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.6</priority>
  </url>`);
  });

  posts.forEach(({ slug, date }) => {
    const lastmod = new Date(date).toISOString();
    urls.push(`  <url>
    <loc>${SITE_URL}/blog/${escapeXml(slug)}/</loc>
    <lastmod>${lastmod}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>`);
  });

  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9
        http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd">
${urls.join('\n')}
</urlset>`;
}

function generateRobotsTxt() {
  return `User-agent: *
Allow: /

Sitemap: ${SITE_URL}/sitemap.xml
`;
}

// ── Main ──────────────────────────────────────────────────────────────────

const posts = getPostsData();

const allTags = Array.from(
  new Set(posts.flatMap(p => p.tags))
).sort();

const allCategories = Array.from(
  new Set(posts.flatMap(p => p.categories))
).sort();

if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

// Один файл со всеми URL
fs.writeFileSync(
  path.join(outDir, 'sitemap.xml'),
  generateSitemap0(posts, allTags, allCategories),
  'utf8'
);
console.log(`✅ sitemap.xml (${posts.length} posts, ${allTags.length} tags, ${allCategories.length} categories)`);

fs.writeFileSync(path.join(outDir, 'robots.txt'), generateRobotsTxt(), 'utf8');
console.log('✅ robots.txt');

console.log(`\n📊 Total: ${posts.length} posts, ${allTags.length} tags, ${allCategories.length} categories`);
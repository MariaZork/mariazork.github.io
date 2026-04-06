import Layout from '@/components/Layout/Layout';
import Image from 'next/image';
import Link from 'next/link';

const interests = [
  'AI in Cybersecurity',
  'AI in Healthcare',
  'Computer Vision',
  'MLOps',
  'Start-ups',
  'LLM',
];

const publications = [
  {
    title: 'Radiation of high-power ultrawideband pulses with elliptical polarization by four-element array of cylindrical helical antennas',
    year: '2018',
    journal: 'Laser and Particle Beams',
    href: '/pdf/laser&particle.pdf',
  },
  {
    title: 'Numerical Modeling of Ultra Wideband Combined Antennas',
    year: '2017',
    journal: 'Russian Physics Journal',
    href: '/pdf/numerical_modeling.pdf',
  },
  {
    title: 'A source of high-power pulses of elliptically polarized ultrawideband radiation',
    year: '2014',
    journal: 'Review of Scientific Instruments',
    href: '/pdf/rsi.pdf',
  },
];

const certifications = [
  {
    title: 'Python for Cybersecurity',
    issuer: 'INFOSEC / Coursera',
    year: '2021',
    href: 'https://www.coursera.org/account/accomplishments/specialization/certificate/9T72NSWJLHVM',
    description: '6-course specialization covering Python for cybersecurity, including scripting, automation, and security tools development.',
  },
  {
    title: 'Machine Learning Engineering for Production (MLOps)',
    issuer: 'DeepLearning.AI',
    year: '2021',
    href: 'https://www.coursera.org/account/accomplishments/specialization/certificate/VVLZLUQGD25L',
    description: '4-course specialization covering ML production, MLOps, CI/CD for ML, and model monitoring.',
  },
  {
    title: 'Data Engineering with Google Cloud',
    issuer: 'Google / Coursera',
    year: '2020',
    href: 'https://www.coursera.org/account/accomplishments/specialization/certificate/79LHSJH25Z74',
    description: '6-course specialization covering GCP data engineering, BigQuery, Dataflow, and ML on GCP.',
  },
  {
    title: 'IBM AI Engineering Specialization',
    issuer: 'IBM / Coursera',
    year: '2020',
    href: 'https://www.coursera.org/account/accomplishments/specialization/XS3WVRQXBC4Z',
    description: '6-course specialization: ML with Python, Spark, Keras, PyTorch, TensorFlow, and a capstone project.',
  },
  {
    title: 'Neural Networks and Computer Vision',
    issuer: 'Samsung R&D / Stepik',
    year: '2019',
    href: 'https://stepik.org/cert/223206',
    description: 'CNNs with PyTorch, challenging math problems, and a Kaggle final assignment.',
  },
  {
    title: 'Introduction to Machine Learning',
    issuer: 'HSE & Yandex / Coursera',
    year: '2018',
    href: 'https://www.coursera.org/account/accomplishments/certificate/3PQD4U7UTPJ6',
    description: 'Rigorous ML course requiring strong mathematical background, by HSE and Yandex School of Data Analysis.',
  },
  {
    title: 'Machine Learning',
    issuer: 'Stanford University / Coursera',
    year: '2017',
    href: 'https://www.coursera.org/account/accomplishments/certificate/7XSCNX4AL85L',
    description: 'The classic Andrew Ng course — all standard ML methods, neural networks, recommender systems.',
  },
];

export default function AboutPage() {
  return (
    <Layout title="About — Maria Zorkaltseva" description="ML Engineer specializing in Computer Vision, NLP, and AI applications.">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-12 md:py-16">

        {/* ── Profile card ── */}
        <div className="bg-white rounded-2xl border border-border shadow-card overflow-hidden mb-10">
          {/* Banner */}
          <div className="h-32 bg-gradient-to-r from-ink via-ink/90 to-primary/80 relative">
            <div
              className="absolute inset-0 opacity-20"
              style={{
                backgroundImage: `radial-gradient(circle at 30% 60%, rgba(26,188,156,0.4) 0%, transparent 60%)`,
              }}
            />
          </div>

          {/* Avatar + name */}
          <div className="px-8 pb-8">
            <div className="relative -mt-14 mb-5 w-24 h-24 rounded-xl overflow-hidden border-4 border-white shadow-card">
              <Image
                src="/images/my_photo.jpg"
                alt="Maria Zorkaltseva"
                fill
                className="object-cover"
                priority
              />
            </div>
            <h1 className="font-serif text-3xl text-ink mb-1">Maria Zorkaltseva</h1>
            <p className="text-primary font-semibold mb-4">Machine Learning Engineer · Paris, France</p>
            <p className="text-ink-muted leading-relaxed max-w-2xl">               
              14+ years of experience in applied AI across healthcare, cybersecurity, IoT/RF sensing. I design and deploy end-to-end ML systems - from research prototyping to production at scale. Author of 19 peer-reviewed publications.<br />
              <br />
              🧩 Current R&amp;D &amp; Projects<br />

              - LLM Engineering - Fine-tuning with LoRA/QLoRA, RAG, LangGraph agent orchestration, MCP integrations<br />
              - Wi-Fi CSI / RF Sensing - Activity recognition and indoor localization<br />
              - Medical Imaging - WSI pathology pipelines: segmentation, tiling, RLE encoding, UNet architectures<br />
              - Physics-Informed ML - PINNs for biomechanical modeling, Koopman operator + Transformer hybrids<br />
              - Data Engineering - Streaming pipelines, BigQuery, dbt, Kestra, dlt<br />
              <br />
              🔬 Research & Industry Interests<br />

              - Edge AI & IoT: on-edge intelligence, distributed IoT systems, sensor-driven applications<br />
              - HealthTech & BioTech: histology, radiology, medical imaging, AI for personalized medicine, drug discovery and investigation, multimodal biological data<br />
              - Multi-Omics & Computational Biology: integrating genomics, transcriptomics, proteomics, and other biological signals<br />
              - Cybersecurity: fraud detection, account takeover prevention, malware detection, adversarial ML<br />
              - LegalTech: NLP for legal documents, semantic search, knowledge extraction<br />
              - Quantitative Engineering: time series modeling, optimization, stochastic calculus, derivative pricing models, portfolio optimization<br />
              - Chess: Stockfish's heuristics, AlphaZero’s neural networks, analytics and AI chess coaching<br />
            </p>

            {/* Social links */}
            <div className="flex flex-wrap gap-3 mt-6">
              <a
                href="https://github.com/MariaZork"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-ink text-white rounded-lg text-sm font-medium hover:bg-ink/80 transition"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
                GitHub
              </a>
              <a
                href="https://www.linkedin.com/in/maria-zorkaltseva/"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-[#0077b5] text-white rounded-lg text-sm font-medium hover:bg-[#006699] transition"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
                LinkedIn
              </a>
              <a
                href="https://medium.com/@maria.zorkaltseva"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-border text-ink rounded-lg text-sm font-medium hover:border-primary/40 hover:text-primary transition"
              >
                Medium Blog
              </a>
              <a
                href="https://scholar.google.com/citations?user=kJHS8ygAAAAJ&hl=ru&authuser=1"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-border text-ink rounded-lg text-sm font-medium hover:border-primary/40 hover:text-primary transition"
              >
                Google Scholar
              </a>
            </div>
          </div>
        </div>

        {/* ── Interests ── */}
        <div className="bg-white rounded-2xl border border-border shadow-card p-8 mb-6">
          <h2 className="font-serif text-2xl text-ink mb-5">My Interests</h2>
          <div className="flex flex-wrap gap-2">
            {interests.map((item) => (
              <span
                key={item}
                className="px-4 py-2 bg-primary-soft text-primary rounded-full text-sm font-medium"
              >
                {item}
              </span>
            ))}
          </div>
        </div>

        {/* ── Publications ── */}
        <div className="bg-white rounded-2xl border border-border shadow-card p-8 mb-6">
          <h2 className="font-serif text-2xl text-ink mb-2">Top Publications</h2>
          <p className="text-ink-muted text-sm mb-6">
            Selected from 19 scientific papers. Full list on{' '}
            <a
              href="https://scholar.google.com/citations?user=kJHS8ygAAAAJ"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              Google Scholar
            </a>.
          </p>
          <div className="space-y-4">
            {publications.map((pub, i) => (
              <div key={i} className="flex gap-4 items-start p-4 rounded-xl bg-surface-alt border border-border">
                <div className="w-10 h-10 rounded-lg bg-primary-soft text-primary flex items-center justify-center text-sm font-bold flex-shrink-0">
                  {pub.year.slice(2)}
                </div>
                <div>
                  <a
                    href={pub.href}
                    className="text-sm font-semibold text-ink hover:text-primary transition block mb-1"
                  >
                    {pub.title}
                  </a>
                  <p className="text-xs text-ink-muted italic">{pub.journal} · {pub.year}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Certifications ── */}
        <div className="bg-white rounded-2xl border border-border shadow-card p-8">
          <h2 className="font-serif text-2xl text-ink mb-6">Certifications</h2>
          <div className="space-y-4">
            {certifications.map((cert, i) => (
              <div key={i} className="flex gap-4 items-start p-4 rounded-xl hover:bg-surface-alt border border-transparent hover:border-border transition-all">
                <div className="w-10 h-10 rounded-lg bg-secondary-soft text-secondary flex items-center justify-center flex-shrink-0">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                  </svg>
                </div>
                <div>
                  <a
                    href={cert.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-semibold text-ink hover:text-primary transition block"
                  >
                    {cert.title}
                  </a>
                  <p className="text-xs text-ink-muted mb-1">{cert.issuer} · {cert.year}</p>
                  <p className="text-xs text-ink-muted leading-relaxed">{cert.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
}

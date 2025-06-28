/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/search',
        destination: 'http://localhost:8000/search', // Replace with your Python backend URL
      },
    ]
  },
}

export default nextConfig

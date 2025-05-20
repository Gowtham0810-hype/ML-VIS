export default function AboutPage() {
  return (
    <div className="container py-12 md:py-24">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl mb-6">About ML-VIS</h1>

        <p className="text-lg mb-6">
          ML-VIS is an educational initiative designed to teach machine learning theory and practical applications
          through interactive visualizations.
        </p>

        <p className="text-lg mb-6">
          Our goal is to make complex machine learning concepts accessible to everyone through visual essays in a fun,
          informative, and engaging manner.
        </p>

        <p className="text-lg mb-6">
          Each visualization is carefully crafted to help you understand the underlying principles of various machine
          learning algorithms, from basic concepts like Linear Regression to more complex techniques like Random Forests
          and Neural Networks.
        </p>

        <h2 className="text-2xl font-bold mt-12 mb-4">Our Approach</h2>

        <p className="text-lg mb-6">
          We believe that interactive learning is one of the most effective ways to understand complex concepts. By
          allowing you to manipulate parameters and see the results in real-time, we help you build an intuitive
          understanding of how these algorithms work.
        </p>

        <p className="text-lg mb-6">Each algorithm page includes:</p>

        <ul className="list-disc list-inside text-lg mb-6 space-y-2">
          <li>An interactive visualization that demonstrates the algorithm in action</li>
          <li>Clear explanations of the underlying principles</li>
          <li>Controls to adjust parameters and see how they affect the results</li>
          <li>Practical examples of where and how the algorithm is used</li>
        </ul>

        <p className="text-lg">We hope that ML-VIS helps you on your journey to understanding machine learning!</p>
      </div>
    </div>
  )
}

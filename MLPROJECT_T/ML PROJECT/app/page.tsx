import Link from "next/link"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ChevronRight } from "lucide-react"

const algorithms = [
  {
    id: "random-forest",
    title: "Random Forest",
    description: "An ensemble learning method that operates by constructing multiple decision trees during training.It works by aggregating the predictions of individual trees.",
    image: "/random-forest.png",
  },
  {
    id: "decision-tree",
    title: "Decision Tree",
    description:
      "A flowchart-like structure where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome.",
    image: "/decision-tree.png",
  },
  {
    id: "linear-regression",
    title: "Linear Regression",
    description:
      "A linear approach to modeling the relationship between a dependent variable and one or more independent variables.",
    image: "/linear-regression.png",
  },
  {
    id: "logistic-regression",
    title: "Logistic Regression",
    description: "Logistic regression is a statistical technique that models the probability of a binary outcome using a logistic function based on input predictor variables.",
    image: "/logistic-regression.jpg",
  },
  {
    id: "knn",
    title: "K-Nearest Neighbors",
    description:
      "A non-parametric method used for classification and regression, where the output is based on the k closest training examples.",
    image: "/knn.jpg",
  },
  {
    id: "svm",
    title: "Support Vector Machine",
    description: "SVM is a powerful supervised learning algorithm used for classification and regression by finding the optimal hyperplane that separates data points.",
    image: "/svm.png",
  },
  {
    id: "dbscan",
    title: "DBSCAN",
    description: "DBSCAN is a density-based clustering algorithm that groups closely packed data points while marking points in low-density regions as outliers or noise.",
    image: "/db-scan.png",
  },
  {
    id: "som",
    title: "Self-Organizing Map",
    description:
      "A type of artificial neural network that is trained using unsupervised learning to produce a low-dimensional representation of the input space.",
    image: "/som.jpeg",
  },
  {
    id: "kmeans",
    title: "K-Means",
    description:
      "An unsupervised machine learning algorithm that groups data points into clusters based on their similarity, where each cluster is defined by a centroid.",
    image: "/k-means.png",
  },
  {
    id: "perceptron",
    title: "Perceptron",
    description:
      "A type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function.",
    image: "/perceptron.jpeg",
  },
]

export default function Home() {
  return (
    <div className="container py-12 md:py-24">
      <div className="mx-auto max-w-3xl text-center mb-16">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
          Visual explanations of core machine learning algorithms
        </h1>
        <p className="mt-6 text-lg text-muted-foreground">
          Interactive visualizations that help you understand how machine learning algorithms work through visual essays
          in a fun, informative, and accessible manner.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {algorithms.map((algorithm) => (
          <Link key={algorithm.id} href={`/algorithms/${algorithm.id}`} className="block">
            <Card className="h-full algorithm-card border border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle>{algorithm.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>{algorithm.description}</CardDescription>
              </CardContent>
              <div className="relative aspect-[16/9] w-full overflow-hidden">
                <img 
                  src={algorithm.image} 
                  alt={algorithm.title} 
                  className="absolute inset-0 w-full h-full  object-center"
                />
              </div>
              <CardFooter className="justify-end">
                <Button variant="ghost" className="gap-1 hover:gap-2 transition-all">
                  Explore <ChevronRight className="h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  )
}
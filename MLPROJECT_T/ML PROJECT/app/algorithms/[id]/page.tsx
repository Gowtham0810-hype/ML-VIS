import { notFound } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ChevronLeft } from "lucide-react"

import LogisticRegressionPage from "@/components/algorithms/logistic-regression"
import LinearRegressionPage from "@/components/algorithms/linear-regression"
import DecisionTreePage from "@/components/algorithms/decision-tree"
import RandomForestPage from "@/components/algorithms/random-forest"
import KNNPage from "@/components/algorithms/knn"
import SVMPage from "@/components/algorithms/svm"
import DBSCANPage from "@/components/algorithms/dbscan"
import SOMPage from "@/components/algorithms/som"
import KMeansPage from "@/components/algorithms/kmeans"
import PerceptronPage from "@/components/algorithms/perceptron"

const algorithms = {
  "logistic-regression": {
    title: "Logistic Regression",
    component: LogisticRegressionPage,
  },
  "linear-regression": {
    title: "Linear Regression",
    component: LinearRegressionPage,
  },
  "decision-tree": {
    title: "Decision Tree",
    component: DecisionTreePage,
  },
  "random-forest": {
    title: "Random Forest",
    component: RandomForestPage,
  },
  knn: {
    title: "K-Nearest Neighbors",
    component: KNNPage,
  },
  svm: {
    title: "Support Vector Machine",
    component: SVMPage,
  },
  dbscan: {
    title: "DBSCAN",
    component: DBSCANPage,
  },
  som: {
    title: "Self-Organizing Map",
    component: SOMPage,
  },
  kmeans: {
    title: "K-Means",
    component: KMeansPage,
  },
  perceptron: {
    title: "Perceptron",
    component: PerceptronPage,
  },
}

export function generateStaticParams() {
  return Object.keys(algorithms).map((id) => ({ id }))
}

export default function AlgorithmPage({ params }: { params: { id: string } }) {
  const algorithm = algorithms[params.id as keyof typeof algorithms]

  if (!algorithm) {
    notFound()
  }

  const AlgorithmComponent = algorithm.component

  return (
    <div className="container py-12">
      <div className="mb-8">
        <Link href="/">
          <Button variant="ghost" className="pl-0 hover:pl-2 transition-all">
            <ChevronLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
        </Link>
      </div>

      <AlgorithmComponent />
    </div>
  )
}

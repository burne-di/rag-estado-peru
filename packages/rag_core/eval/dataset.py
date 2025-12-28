"""
Dataset de evaluación para RAG
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EvalItem:
    """Un item de evaluación"""

    question: str
    expected_sources: list[str]  # IDs de fuentes esperadas
    gold_answer: Optional[str] = None  # Respuesta esperada (opcional)
    category: Optional[str] = None  # Categoría de la pregunta
    difficulty: Optional[str] = None  # easy, medium, hard

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalItem":
        return cls(**data)


class EvalDataset:
    """Dataset de evaluación con carga/guardado JSONL"""

    def __init__(self, items: list[EvalItem] = None):
        self.items = items or []

    def add(self, item: EvalItem):
        """Agrega un item al dataset"""
        self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def save(self, path: str | Path):
        """Guarda dataset en formato JSONL"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "EvalDataset":
        """Carga dataset desde JSONL"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {path}")

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    items.append(EvalItem.from_dict(data))

        return cls(items)

    @classmethod
    def create_sample(cls) -> "EvalDataset":
        """Crea un dataset de ejemplo para desarrollo"""
        items = [
            EvalItem(
                question="¿Qué es el Código Tributario?",
                expected_sources=["Codigo-Tributario-Sunat.pdf"],
                gold_answer="El Código Tributario es el marco legal que regula las relaciones entre el contribuyente y la Administración Tributaria.",
                category="definicion",
                difficulty="easy",
            ),
            EvalItem(
                question="¿Cuál es el plazo para presentar una reclamación tributaria?",
                expected_sources=["Codigo-Tributario-Sunat.pdf"],
                gold_answer="El plazo para presentar una reclamación tributaria es de 20 días hábiles.",
                category="plazos",
                difficulty="medium",
            ),
            EvalItem(
                question="¿Qué obligaciones tienen los contribuyentes?",
                expected_sources=["Codigo-Tributario-Sunat.pdf"],
                category="obligaciones",
                difficulty="medium",
            ),
            EvalItem(
                question="¿Cuándo prescribe la acción para determinar la obligación tributaria?",
                expected_sources=["Codigo-Tributario-Sunat.pdf"],
                category="prescripcion",
                difficulty="hard",
            ),
            EvalItem(
                question="¿Qué es la SUNAT?",
                expected_sources=["Codigo-Tributario-Sunat.pdf"],
                category="instituciones",
                difficulty="easy",
            ),
        ]
        return cls(items)

    def filter_by_category(self, category: str) -> "EvalDataset":
        """Filtra items por categoría"""
        filtered = [item for item in self.items if item.category == category]
        return EvalDataset(filtered)

    def filter_by_difficulty(self, difficulty: str) -> "EvalDataset":
        """Filtra items por dificultad"""
        filtered = [item for item in self.items if item.difficulty == difficulty]
        return EvalDataset(filtered)

    def get_categories(self) -> list[str]:
        """Obtiene todas las categorías únicas"""
        return list(set(item.category for item in self.items if item.category))

    def get_stats(self) -> dict:
        """Estadísticas del dataset"""
        categories = {}
        difficulties = {}

        for item in self.items:
            if item.category:
                categories[item.category] = categories.get(item.category, 0) + 1
            if item.difficulty:
                difficulties[item.difficulty] = difficulties.get(item.difficulty, 0) + 1

        return {
            "total_items": len(self.items),
            "with_gold_answer": sum(1 for i in self.items if i.gold_answer),
            "categories": categories,
            "difficulties": difficulties,
        }

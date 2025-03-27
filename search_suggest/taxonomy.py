"""
Taxonomy parser for Google Merchant Categories.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


class TaxonomyParser:
    """Parser for Google Merchant Category taxonomy files."""

    def __init__(self, taxonomy_file: Path):
        """Initialize the taxonomy parser.

        Args:
            taxonomy_file: Path to the taxonomy file
        """
        self.taxonomy_file = taxonomy_file
        self.categories: Dict[str, Dict] = {}
        self._parse_taxonomy()
        self._build_category_hierarchy()

    def _parse_taxonomy(self) -> None:
        """Parse the taxonomy file and build the category hierarchy."""
        with open(self.taxonomy_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse the category line
                try:
                    id_part, category_part = line.split(" - ", 1)
                    category_id = id_part.strip()
                    category_path = category_part.strip()
                    
                    # Split the category path into individual categories
                    categories = category_path.split(" > ")
                    
                    # Store the category with its full path and ID
                    self.categories[category_id] = {
                        "id": category_id,
                        "full_path": category_path,
                        "path_parts": categories,
                        "level": len(categories),
                        "name": categories[-1]
                    }
                except ValueError:
                    # Skip malformed lines
                    continue

    def _build_category_hierarchy(self) -> None:
        """Build a hierarchy of categories for efficient lookup of subcategories."""
        # Create a mapping of category paths to their IDs
        path_to_ids: Dict[str, List[str]] = {}
        for cat_id, cat_info in self.categories.items():
            path = cat_info["full_path"]
            path_parts = cat_info["path_parts"]
            
            # For each prefix of the path, add this category as a subcategory
            for i in range(1, len(path_parts) + 1):
                prefix = " > ".join(path_parts[:i])
                if prefix not in path_to_ids:
                    path_to_ids[prefix] = []
                path_to_ids[prefix].append(cat_id)
        
        # Add subcategory IDs to each category
        for cat_id, cat_info in self.categories.items():
            path = cat_info["full_path"]
            self.categories[cat_id]["subcategory_ids"] = path_to_ids.get(path, [])

    def get_categories_up_to_level(self, max_level: int = 3) -> List[Dict]:
        """Get all categories up to the specified level.

        Args:
            max_level: Maximum level of categories to include

        Returns:
            List of category dictionaries
        """
        return [cat for cat in self.categories.values() if cat["level"] <= max_level]

    def get_subcategory_names(self, category_id: str) -> List[str]:
        """Get the names of all subcategories for a given category.

        Args:
            category_id: ID of the category

        Returns:
            List of subcategory names
        """
        category = self.categories.get(category_id)
        if not category:
            return []
        
        # Get all subcategory IDs
        subcategory_ids = category.get("subcategory_ids", [])
        
        # Get the names of all subcategories
        subcategory_names = []
        for sub_id in subcategory_ids:
            sub_category = self.categories.get(sub_id)
            if sub_category and sub_id != category_id:  # Exclude self
                subcategory_names.append(sub_category["name"])
        
        return subcategory_names

    def create_rich_category_text(self, category_id: str) -> str:
        """Create rich text for a category by combining it with its subcategories.

        Args:
            category_id: ID of the category

        Returns:
            Rich text representation of the category
        """
        category = self.categories.get(category_id)
        if not category:
            return ""
        
        # Start with the full path
        rich_text = category["full_path"]
        
        # Add subcategory names
        subcategory_names = self.get_subcategory_names(category_id)
        if subcategory_names:
            rich_text += " " + " ".join(subcategory_names)
        
        return rich_text

    def get_rich_categories_for_embedding(self, max_level: int = 3) -> List[Tuple[str, str]]:
        """Get categories with rich text for embedding.

        Args:
            max_level: Maximum level of categories to include

        Returns:
            List of tuples with (category_id, rich_text)
        """
        categories = self.get_categories_up_to_level(max_level)
        return [(cat["id"], self.create_rich_category_text(cat["id"])) for cat in categories]

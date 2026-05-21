"""
Database module for managing people data
"""

import json
import os
from pathlib import Path
from config import DB_FILE, FACES_DIR


class PeopleDatabase:
    """Manages the people database and face samples storage"""
    
    def __init__(self):
        self.db_file = DB_FILE
        self.faces_dir = FACES_DIR
        self._ensure_directories()
        self.data = self._load()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        Path(self.faces_dir).mkdir(exist_ok=True)
    
    def _load(self):
        """Load database from JSON file"""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save(self):
        """Save database to JSON file"""
        with open(self.db_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_next_id(self):
        """Get the next available person ID"""
        if not self.data:
            return 0
        return max(int(k) for k in self.data.keys()) + 1
    
    def add_person(self, name, person_id):
        """Add a new person to the database"""
        internal_id = self.get_next_id()
        self.data[str(internal_id)] = {
            "name": name,
            "id": person_id
        }
        self._save()
        return internal_id
    
    def get_person(self, internal_id):
        """Get person info by internal ID"""
        return self.data.get(str(internal_id))
    
    def get_all_people(self):
        """Get all registered people"""
        return self.data
    
    def get_person_dir(self, internal_id):
        """Get the directory path for a person's face samples"""
        return os.path.join(self.faces_dir, str(internal_id))
    
    def create_person_dir(self, internal_id):
        """Create directory for a person's face samples"""
        person_dir = self.get_person_dir(internal_id)
        Path(person_dir).mkdir(exist_ok=True)
        return person_dir
    
    def get_face_samples(self, internal_id):
        """Get all face sample paths for a person"""
        person_dir = self.get_person_dir(internal_id)
        if not os.path.exists(person_dir):
            return []
        
        samples = []
        for file in os.listdir(person_dir):
            if file.endswith('.jpg'):
                samples.append(os.path.join(person_dir, file))
        return samples
    
    def count(self):
        """Get total number of registered people"""
        return len(self.data)

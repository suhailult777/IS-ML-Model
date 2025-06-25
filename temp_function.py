    def _extract_relevant_content(self, text: str, question: str) -> str:
        """Extract the most relevant part of the text for the question"""
        import re
        
        # Clean the text more thoroughly
        text = text.strip()
        
        # Fix common OCR/PDF extraction issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces between words
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)  # Space between word and number
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)  # Space between number and word
        
        # Find the specific section about diagnosis/tests if that's what's being asked
        question_lower = question.lower()
        if 'diagnos' in question_lower or 'test' in question_lower:
            # Look for the diagnosis section specifically
            diagnosis_match = re.search(r'The tests most commonly used to diagnos[ie]s?\s*[A-Z]?[A-Z]?I?\s*include:\s*([^.]+(?:\.[^.]+)*)', text, re.IGNORECASE | re.DOTALL)
            if diagnosis_match:
                diagnosis_text = diagnosis_match.group(0)
                # Clean up the extracted text
                diagnosis_text = re.sub(r'\s+', ' ', diagnosis_text)
                # Complete the sentence properly
                if not diagnosis_text.endswith('.'):
                    diagnosis_text += '.'
                return diagnosis_text
        
        # If text is short enough, return it as is (but cleaned)
        if len(text) <= 500:
            return text
        
        # Split into sentences and find relevant ones
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 20]
        
        # Find keywords from question
        question_keywords = [word for word in question_lower.split() 
                           if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where', 'tell', 'about']]
        
        # Score sentences based on keyword matches
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in question_keywords if keyword in sentence.lower())
            if score > 0:
                scored_sentences.append((sentence, score))
        
        if scored_sentences:
            # Sort by score and take best ones
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentences = [s[0] for s in scored_sentences[:3]]
            result = '. '.join(best_sentences)
            if not result.endswith('.'):
                result += '.'
            return result
        
        # If no keyword matches, return first part of text
        return sentences[0] + '.' if sentences else text[:500]

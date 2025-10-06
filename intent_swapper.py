import json

class IntentGatheringTool:
    def __init__(self, data_path):
        """
        Initialize the tool with the path to the dataset.
        
        Args:
            data_path (str): Path to the JSON data file containing intents and utterances.
        """
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        """Load the data from the provided JSON file."""
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        return data

    def list_intents(self):
        """List all unique intents."""
        intents = set()
        for dialogue in self.data:
            for turn in dialogue['turns']:
                intents.add(turn['intent'])
        return sorted(list(intents))

    def search_intent(self, keyword):
        """
        Search for intents and show their utterances based on a keyword.
        
        Args:
            keyword (str): The keyword to search for in the intent headings and utterances.
        
        Returns:
            List of matching intents and their corresponding utterances.
        """
        matches = []
        for dialogue in self.data:
            for turn in dialogue['turns']:
                if keyword.lower() in turn['intent'].lower():
                    matches.append((turn['intent'], turn['utterance']))

        return matches

    def display_intents_and_utterances(self, matches):
        """
        Display intents and their corresponding utterances.

        Args:
            matches (list): A list of tuples (intent, utterance) that match the search.
        """
        if matches:
            current_intent = ""
            for intent, utterance in matches:
                if intent != current_intent:
                    current_intent = intent
                    print(f"\nIntent: {intent}")
                print(f"  - Utterance: {utterance}")
        else:
            print("No matches found.")

    def replace_intent(self, old_intent, new_intent):
        """
        Replace all occurrences of an old intent with a new one.
        
        Args:
            old_intent (str): The intent to be replaced.
            new_intent (str): The new intent to replace it with.
        """
        for dialogue in self.data:
            for turn in dialogue['turns']:
                if turn['intent'] == old_intent:
                    turn['intent'] = new_intent

        print(f"Replaced all occurrences of '{old_intent}' with '{new_intent}'.")

    def save_data(self, output_path):
        """Save the modified dataset to a JSON file."""
        with open(output_path, 'w') as file:
            json.dump(self.data, file, indent=4)
        print(f"Modified data saved to {output_path}.")

    def run(self):
        """Main interactive loop for the tool."""
        while True:
            print("\n--- Intent Gathering Tool ---")
            print("1. Show all intents")
            print("2. Search for an intent and display utterances")
            print("3. Replace an intent")
            print("4. Save and exit")
            choice = input("\nEnter your choice (1-4): ")

            if choice == '1':
                # Show all intents
                all_intents = self.list_intents()
                print("\nAll Intents:")
                for intent in all_intents:
                    print(f"- {intent}")

            elif choice == '2':
                # Search for an intent and show utterances
                keyword = input("\nEnter keyword to search for: ")
                matches = self.search_intent(keyword)
                self.display_intents_and_utterances(matches)

            elif choice == '3':
                # Replace an intent
                old_intent = input("\nEnter the intent you want to replace: ")
                new_intent = input("Enter the new intent: ")
                self.replace_intent(old_intent, new_intent)

            elif choice == '4':
                # Save and exit
                output_path = input("\nEnter output file path (e.g., modified_data.json): ")
                self.save_data(output_path)
                break

            else:
                print("Invalid choice. Please select an option between 1 and 4.")

# Example usage
if __name__ == "__main__":
    data_path = 'C:\\Users\\Ashwin Rajhans\\SIH\\SIH\\data.json'  # Update with your dataset path
    tool = IntentGatheringTool(data_path)
    tool.run()


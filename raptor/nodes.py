class SummaryNode:
    def __init__(self, text, children=None):
        self.text = text
        self.children = children if children else []

    def add_child(self, child):
        self.children.append(child)

    def get_text(self):
        return self.text

    def get_children(self):
        return self.children

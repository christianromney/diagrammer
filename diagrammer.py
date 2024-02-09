from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import sys, select

class DataFlowDiagrammer:
  def __init__(self, model=ChatOpenAI(model="gpt-4-turbo-preview"), parser=StrOutputParser(),
               formatter_prompt="formatter-prompt.txt", diagrammer_prompt="diagrammer-prompt.txt"):
    self.model = model
    self.parser = parser

  def create_specification(self, description):
    """Convert a natural language description to a data flow 'specification' an
    LLM has been prompted to understand."""
    format_instructions = open(self.formatter_prompt, "r")
    spec_prompt =  ChatPromptTemplate.from_messages([
      ("system", format_instructions.read()),
      ("user", "Convert the following data flow description to an interaction list: {description}")
    ])
    spec_chain = spec_prompt | self.model | self.parser
    return spec_chain.invoke({"description": description})

  def generate_dot_diagram(self, spec):
    "Generate a diagram from a (bulleted list) interaction specification."
    diagram_instructions = open(self.diagrammer_prompt, "r")
    diagram_prompt = ChatPromptTemplate.from_messages([
      ("system", diagram_instructions.read()),
      ("user", "Diagram the following data flow: {spec}")
    ])
    diagram_chain = diagram_prompt | self.model | self.parser
    return diagram_chain.invoke({"spec": spec})

  def invoke(self, description):
    "Generate a Data Flow diagram from a description."
    spec = self.create_specification(description)
    return self.generate_dot_diagram(spec)

# helper function
def read_stdin():
  "Read a string from stdin to EOF. Throws an error if no input is provided."
  timeout = 0 # Non-blocking check
  # check if there is input available on stdin
  ready, _, _ = select.select([sys.stdin], [], [], timeout)
  if ready:
    return sys.stdin.read().strip()
  else:
    raise RuntimeError("No input provided on stdin. Please provide input and try again.")

if __name__ == "__main__":
  description = read_stdin()
  diagram = DataFlowDiagrammer().invoke(description)
  print(diagram)

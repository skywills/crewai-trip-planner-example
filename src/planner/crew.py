from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from planner.tools.browser_tools import BrowserTools
from planner.tools.calculator_tools import CalculatorTools
from planner.tools.search_tools import SearchTools

from langchain_openai import ChatOpenAI
import os
#os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model = os.environ["OPENAI_MODEL_NAME"],
    base_url = os.environ["OPENAI_API_BASE"])

@CrewBase
class TripCrew:
  """Trip planner crew"""

  agents_config = "config/agents.yaml"
  tasks_config = "config/tasks.yaml"

  @agent
  def city_selector_agent(self) -> Agent:
    return Agent(
      config=self.agents_config["city_selection_agent"],
      tools=[
        SearchTools.search_internet,
        BrowserTools.scrape_and_summarize_website,
      ],
      llm = llm
    )
  
  @agent
  def local_expert(self) -> Agent:
    return Agent(
      config=self.agents_config["local_expert"],
      tools=[
        SearchTools.search_internet,
        BrowserTools.scrape_and_summarize_website,
      ],
      llm = llm
    )
  
  @agent
  def travel_concierge(self) -> Agent:
    return Agent(
      config=self.agents_config["travel_concierge"],
      tools=[
        SearchTools.search_internet,
        BrowserTools.scrape_and_summarize_website,
        CalculatorTools.calculate,
      ],
      llm = llm
    )
  
  @task
  def identify_task(self) -> Task:
    return Task(
      config=self.tasks_config["identify"],
      agent=self.city_selector_agent(),
    )
  
  @task
  def gather_task(self) -> Task:
    return Task(
      config=self.tasks_config["gather_info"],
      agent=self.local_expert(),
    )
  
  @task
  def trip_planning_task(self) -> Task:
    return Task(
      config=self.tasks_config["trip_planning"],
      agent=self.travel_concierge(),
    )
  
  @crew
  def crew(self) -> Crew:
    return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
      process=Process.sequential,
      verbose=2,
    )
#!/usr/bin/env python
from crew import TripCrew
import datetime
from textwrap import dedent

def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'location': input(
            dedent("""
              From where will you be traveling from?
            """)),
        'cities': input(
            dedent("""
              What are the cities options you are interested in visiting?
            """)),
        'date_range':  input(
            dedent("""
              What is the date range you are interested in traveling?
            """)),
        'interests': input(
            dedent("""
              What are some of your high level interests and hobbies?
            """))
    }
    result = TripCrew().crew().kickoff(inputs=inputs)
    print("\n\n########################")
    print("## Here is you Trip Plan")
    print("########################\n")
    print(result)    
from langchain_core.retrievers import BaseRetriever
from langchain_google_vertexai import VertexAI
from langchain_core.documents import Document
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Import custom Matching Engine packages
from utils.matching_engine import MatchingEngine

# Neo4j Helper scripts
from utils.NEO4J_Graph import Graph


class FHIRResourcesRetriever(BaseRetriever):
    
    neo4j_graph: Graph
    me: MatchingEngine
    llm: VertexAI
    
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # In a real scenario, you'd fetch and process FHIR data here
        resource_type = self.get_resource_type(query)
        patient_name = self.get_patient_name(query)
        patient_id = self.get_patient_id(patient_name)
        resources = self.get_patient_resources(query,resource_type, patient_id)
        
        # For this example, we simply return our "Hello World" document
        documents = [Document(page_content=f"The Patient name is {patient_name} \nFHIR Resource Type is {resource_type} \nPatient_ID={patient_id} \nBelow is the medical information of {patient_name}: \n{resources}")]
        return documents
    
    # Get Patient Name
    def get_patient_name(self, query):
        _response = self.llm(f'''
            system:Given the following question from the user, identify all potential first and last names within this sentence.

            The name might contain numbers e.g. Carmelo33, Reichert620, Antone63
            The name might also contain numbers e.g. Andrea7, Jenkins714, Chasity985, Pagac496, Carmelo33
            The name might also contain Apostrophe e.g. Andrea's, John's, Johns' James'


            Return the answer formatted as first-name last-name.

            Use the form:
            first-name last name

            Please do not include any special formatting characters, like new lines or "\\n".
            Please do not include triple quotes.

            If there are no names, do not make one up. 
            If there are no names return None""

            user:{query}
            ''')

        names = _response
        names = names.strip()

        if names == 'None':
            names = input("Could you please help me with the Patient name:")
        return names
    
    # Get Patient ID
    def get_patient_id(self, patient_name: str):
        
        # Reconstruct the Query such that it matches with Resource Text stored in Vector Search to improve retrieval accuracy
        patient_query = f"""
        The type of information in this entry is patient. 
        The name use for this patient is official. The name family for this patient{patient_name}
        The name given 0 for this patient is {patient_name}
        """
        
        response = self.me.similarity_search(patient_query, k=1, search_distance=.8)
        patient_id = response[0].metadata['fhir_patient_id']
        return patient_id
        
    
    # Get Resource Type
    def get_resource_type(self, query):

        _response = self.llm(f'''
        system:
        FHIR (Fast Healthcare Interoperability Resources) is a standard for exchanging healthcare data. 
        It defines various resources (like Patient, Observation, Procedure) to represent clinical concepts.  
        See all resource types here:  https://hl7.org/fhir/DSTU2/resourcelist.html

        Given the following user's natural language question about healthcare data, help me understand which FHIR resources might be relevant

        The output must be list of FHIR Resource Types.

        If you are unable to identify Resource Types from the user question, do not make one up. 
        If you are unable to identify Resource Types from the user question return an empty string link

        Examples
        - Question: Where can I find a patient's immunization history?
          Possible FHIR Resource Types: Patient (for demographics), Immunization (to record immunization events)
        - Question: How do I track medication dosage changes?
          Possible FHIR Resource Types: MedicationRequest, MedicationDispense (depends on dosage tracking detail needed)
        - Question: What can you tell me about Alina705's claim created on 03/17/2007?
          Possible FHIR Resource Types: Claim, Claim Response
        - Question: Explain why a Throat culture procedure was performed on Antone63 on 2014-04-20?
          Possible FHIR Resource Types: Procedure
        - Question: What medications is Antone69 allergic to?
          Possible FHIR Resource Types: AllergyIntolerance
        - Question: What is Babara869's height?
          Possible FHIR Resource Types: Observation
        - Question: When was the last blood pressure reading for Carmelo33 taken?
          Possible FHIR Resource Types: Observation
        - Question: Based on this explanation of benefits created on February 11, 1999, how much did it cost and what service was provided?
          Possible FHIR Resource Types: ExplanationOfBenefit
        ""

        user:{query}
        ''')

        fhir_resource_type = _response
        return fhir_resource_type
    
    # Fetch Patient's Resources relevant to use query
    def get_patient_resources(self, query: str, resource_type:str, patient_id: str):
        resource_query = f"The type of information in this entry is {resource_type}. {query}"        
        filters = {"namespace": "fhir_patient_id", "allow_list": [patient_id]}
        responses = self.me.similarity_search(resource_query, k=20, search_distance=.75, filters=filters)
        return responses

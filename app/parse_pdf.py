# python
import json
import os

# pdf parsing
import camelot
import PyPDF2


def extract_tables():
    '''
    example of how to use camelot to parse tables out of pdfs
    '''

    tables = camelot.read_pdf(data_path, pages="17", flavor="stream")
    tables[0].to_csv("/results/results2.csv")


def extract_text():
    
    # todo capture potential error
    pdf_path = os.environ["PDF_PATH"]
    pdf_results_path = os.environ["PDF_RESULT_PATH"]

    with open(pdf_path, 'rb') as pdf_file_obj:

        pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)

        # printing number of pages in pdf file 
        print("Found {} pages".format(pdf_reader.numPages)) 
        
        # extracting text from pages 
        page_text = []
        page_text.append("page,text")
        for page_index in range(pdf_reader.numPages):

            # creating a page object 
            page_obj = pdf_reader.getPage(page_index)

            # extract text and cleanup for csv
            extracted_text = page_obj.extractText().replace("\n", " ").replace(",", " ")

            page_text.append("{}, {}".format(page_index, extracted_text))

        #saving csv
        with open("/results/parsed_pdf.csv", "wb") as fh:

            formatted_text = "\n".join(page_text)
            fh.write(formatted_text.encode(encoding='utf-8', errors="ignore"))



if __name__ == "__main__":
    extract_text()
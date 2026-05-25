import os
import fitz

def generate_pdf():
    # Define paths
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    pdf_path = os.path.join(raw_dir, "Placement_RAG_Dataset_Enhanced.pdf")

    # Create new document
    doc = fitz.open()

    # --- PAGE 1: Section 1 (Eligibility Table) ---
    page = doc.new_page()
    page.insert_text((50, 50), "Section 1: Company Eligibility Profiles", fontsize=18, fontname="hebo")
    page.insert_text((50, 75), "Official requirements and cutoffs for major recruiters.", fontsize=11, fontname="helv")

    # Table headers
    headers = ["Company", "Min CGPA", "Max Backlogs", "Package (LPA)", "Bond (Yrs)", "Key Topics", "Tech Focus"]
    start_y = 110
    col_x = [50, 120, 180, 250, 320, 380, 480]

    for col_idx, header in enumerate(headers):
        page.insert_text((col_x[col_idx], start_y), header, fontsize=10, fontname="hebo")
    
    # Draw header line
    page.draw_line((50, start_y + 5), (550, start_y + 5), width=1)

    # Table rows formatted for regex fallback
    rows = [
        ["Google", "7.4", "0", "42.0", "1", "DSA, Algorithms", "Python"],
        ["Amazon", "6.4", "0", "32.0", "0", "DSA, System Design", "Java"],
        ["TCS", "6.5", "2", "4.5", "2", "Aptitude, Basic Coding", "C++"],
        ["Infosys", "6.5", "2", "4.0", "2", "Aptitude, Web Dev", "Python"],
        ["Deloitte", "6.5", "1", "6.5", "1", "SQL, Communication", "Java"],
        ["Microsoft", "8.0", "0", "51.0", "0", "DSA, System Design", "C++"],
        ["Accenture", "6.5", "2", "4.5", "1", "OOP, Communication", "Java"],
        ["Wipro", "6.0", "3", "3.5", "2", "Coding, Aptitude", "Java"],
        ["Cognizant", "6.0", "2", "4.0", "2", "Coding, Communication", "Python"],
        ["Capgemini", "6.0", "2", "4.0", "2", "Coding, Communication", "Java"],
        ["Intel", "7.5", "0", "32.5", "0", "DSA, Hardware", "Python"]
    ]

    curr_y = start_y + 20
    for r in rows:
        # Write fields separated by standard spacing for regex matching
        line_str = f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]}"
        page.insert_text((50, curr_y), line_str, fontsize=9, fontname="helv")
        curr_y += 20

    # --- PAGE 2: Section 2 (Interview Experiences - with duplicates) ---
    page = doc.new_page()
    page.insert_text((50, 50), "Section 2: Interview Experiences", fontsize=18, fontname="hebo")
    page.insert_text((50, 75), "Student interview experience notes.", fontsize=11, fontname="helv")

    experiences = [
        "■ TCS | Technical Focus: Java, C++",
        "Round 1 Details: Online assessment containing DSA coding questions.",
        "Round 1 Details: Online assessment containing DSA coding questions.",
        "Round 1 Details: Online assessment containing DSA coding questions.",
        "Round 2 Details: Technical round testing OOP concepts and basic database queries.",
        "Tip: Master fundamentals of OOP, arrays, and string manipulations.",
        "Tip: Master fundamentals of OOP, arrays, and string manipulations.",
        "",
        "■ Amazon | Technical Focus: Java, System Design",
        "Round 1 Details: Online assessment with two DSA coding questions.",
        "Round 1 Details: Online assessment with two DSA coding questions.",
        "Round 2 Details: Core technical round focusing heavily on tree algorithms and system scalability.",
        "Round 2 Details: Core technical round focusing heavily on tree algorithms and system scalability.",
        "Tip: Be ready for high scale system design discussions.",
        "Tip: Be ready for high scale system design discussions.",
        "Tip: Be ready for high scale system design discussions."
    ]

    curr_y = 110
    for line in experiences:
        page.insert_text((50, curr_y), line, fontsize=10, fontname="hebo" if line.startswith("■") else "helv")
        curr_y += 20

    # --- PAGE 3: Section 3 (Hiring Distribution Table for fallback comparisons) ---
    page = doc.new_page()
    page.insert_text((50, 50), "Section 3: Hiring Distribution by Role", fontsize=18, fontname="hebo")
    page.insert_text((50, 75), "Distribution of offers by roles across different recruiters.", fontsize=11, fontname="helv")

    hiring_headers = ["Company", "SDE", "Analyst", "Officer", "Intern"]
    start_y = 110
    col_x_hiring = [50, 150, 220, 290, 360]

    for col_idx, header in enumerate(hiring_headers):
        page.insert_text((col_x_hiring[col_idx], start_y), header, fontsize=10, fontname="hebo")
    page.draw_line((50, start_y + 5), (420, start_y + 5), width=1)

    hiring_rows = [
        ["TCS", "88", "42", "70", "44"],
        ["Infosys", "30", "68", "62", "22"],
        ["Amazon", "42", "36", "40", "82"],
        ["Google", "30", "92", "46", "30"],
        ["Microsoft", "58", "58", "36", "68"]
    ]

    curr_y = start_y + 20
    for r in hiring_rows:
        line_str = f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]}"
        page.insert_text((50, curr_y), line_str, fontsize=9, fontname="helv")
        curr_y += 20

    # --- PAGE 4: Section 5 (Temporal Trends) ---
    page = doc.new_page()
    page.insert_text((50, 50), "Section 5: Temporal Trends", fontsize=18, fontname="hebo")
    page.insert_text((50, 75), "Placement hiring trends across academic seasons.", fontsize=11, fontname="helv")

    trends_narratives = [
        "In the 2024 placement season, TCS saw a steady demand with an emphasis on cloud skills.",
        "In the 2025 placement season, Google increased its hiring for machine learning engineering roles.",
        "In the 2026 placement season, Microsoft hired heavily for software engineering positions.",
        "In the 2023 placement season, Amazon focused on expanding cloud infrastructure talent."
    ]

    curr_y = 110
    for line in trends_narratives:
        page.insert_text((50, curr_y), line, fontsize=10, fontname="helv")
        curr_y += 30

    # --- PAGE 5: Section 6 (Conflicting Requirements) ---
    page = doc.new_page()
    page.insert_text((50, 50), "Section 6: Conflicting Requirements", fontsize=18, fontname="hebo")
    page.insert_text((50, 75), "Contains conflicting portal information vs official placement cell criteria.", fontsize=11, fontname="helv")

    conflicts = [
        "Amazon Official CGPA cutoff is 6.4 (Source: official)",
        "Amazon Portal CGPA cutoff is 7.0 (Source: portal)",
        "Google Official package is 42.0 (Source: official)",
        "Google Portal package is 45.0 (Source: portal)"
    ]

    curr_y = 110
    for line in conflicts:
        page.insert_text((50, curr_y), line, fontsize=10, fontname="helv")
        curr_y += 25

    # --- PAGE 6: Section 7 (Overall Placement Statistics) ---
    page = doc.new_page()
    page.insert_text((50, 50), "Section 7: Overall Placement Statistics", fontsize=18, fontname="hebo")
    page.insert_text((50, 75), "Overall company average packages and service agreement statistics.", fontsize=11, fontname="helv")

    stats_headers = ["Company", "Average Package", "Bond Details", "Placement Rate"]
    start_y = 110
    col_x_stats = [50, 150, 260, 360]

    for col_idx, header in enumerate(stats_headers):
        page_s = page
        page.insert_text((col_x_stats[col_idx], start_y), header, fontsize=10, fontname="hebo")
    page.draw_line((50, start_y + 5), (460, start_y + 5), width=1)

    stats_rows = [
        ["Cognizant", "34.2", "2", "92%"],
        ["Intel", "32.5", "0", "88%"],
        ["Google", "42.0", "1", "95%"],
        ["Microsoft", "51.0", "0", "94%"],
        ["Amazon", "32.0", "0", "90%"],
        ["TCS", "4.5", "2", "85%"],
        ["Infosys", "4.0", "2", "82%"],
        ["Deloitte", "6.5", "1", "80%"],
        ["Accenture", "4.5", "1", "85%"],
        ["Wipro", "3.5", "2", "78%"],
        ["Capgemini", "4.0", "2", "75%"]
    ]

    curr_y = start_y + 20
    for r in stats_rows:
        line_str = f"{r[0]} | {r[1]} | {r[2]} | {r[3]}"
        page.insert_text((50, curr_y), line_str, fontsize=9, fontname="helv")
        curr_y += 20

    # Save document
    doc.save(pdf_path)
    doc.close()
    print(f"Generated PDF dataset successfully at {pdf_path}")

    # Also output overall_placement_statistics_section7.pdf as requested
    stats_pdf_path = os.path.join(raw_dir, "overall_placement_statistics_section7.pdf")
    doc_stats = fitz.open()
    page_s = doc_stats.new_page()
    page_s.insert_text((50, 50), "Section 7: Overall Placement Statistics", fontsize=18, fontname="hebo")
    page_s.insert_text((50, 75), "Overall company average packages and service agreement statistics.", fontsize=11, fontname="helv")
    
    for col_idx, header in enumerate(stats_headers):
        page_s.insert_text((col_x_stats[col_idx], start_y), header, fontsize=10, fontname="hebo")
    page_s.draw_line((50, start_y + 5), (460, start_y + 5), width=1)
    
    curr_y = start_y + 20
    for r in stats_rows:
        line_str = f"{r[0]} | {r[1]} | {r[2]} | {r[3]}"
        page_s.insert_text((50, curr_y), line_str, fontsize=9, fontname="helv")
        curr_y += 20
        
    doc_stats.save(stats_pdf_path)
    doc_stats.close()
    print(f"Generated separate statistics PDF successfully at {stats_pdf_path}")

if __name__ == "__main__":
    generate_pdf()

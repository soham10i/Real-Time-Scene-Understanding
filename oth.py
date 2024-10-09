def combine_lecture_details(professor, subject, semester, time, room):
    return "Professor " + professor + " will lecture subject " + subject + " in the " + semester + " at " + time + " in Room " + room
    
def get_exam_result_in_percentage(total_points, student_points):
    return round(student_points/total_points*100, 2)
    
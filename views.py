from flask import Flask
from flask import render_template, request, redirect, url_for
from time import time
from methods import *

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploaded/'


@app.route('/')
def load_homepage():
    return redirect(url_for('home'))


@app.route('/home')
def home():
    return render_template("home.html")


@app.route('/home', methods=['POST'])
def upload():
    global labeled
    global unlabeled

    if request.form['labeled_file'] == '' or request.form['k'] == '' or request.form['threshold'] == '':
        return 'Error: no input'
    print(request.form['unlabeled_file'] == '')
    # default unlabeled sentences file
    if request.form['unlabeled_file'] == '':
        with open('static/unlabeled_default.txt', 'r') as file:
            unlabeled = parse_unlabeled_sentences_text(file.read())
    else:
        unlabeled = parse_unlabeled_sentences_text(request.form['unlabeled_file'])

    labeled = parse_labeled_sentences_text(request.form['labeled_file'])
    k = int(request.form['k'])
    threshold = float(request.form['threshold'])
    results = '\n\n'.join(run(k, threshold))
    return results


def run(k, threshold):
    """
    The main function.
    Returns a list of newly labeled sentences, represented as JSON files
    """
    start_time = time()
    # labeled = parse_labeled_sentences_zip()
    # unlabeled = parse_unlabeled_sentences()
    for s in unlabeled:
        s.print_sentence()
    # for every unlabeled sentence u, find maximal alignment with some labeled sentence l
    for u in unlabeled:
        targets = get_target_predicates(u)

        for t in targets:
            max_score = 0
            max_l = None
            max_alignment = None

            for l in relevant_seeds(labeled, t, threshold):
                alignment_domain, paths = build_domain(l)
                alignment_range = build_range(u, t, paths)
                alignment, score = find_optimal_alignment(alignment_domain, alignment_range)

                if score > max_score and complete(alignment, alignment_domain):
                    max_l = l
                    max_alignment = alignment
                    max_score = score
                    """
                    # debug
                    print_domain(l, alignment_domain)
                    print_range(u, alignment_range)
                    print_alignment(alignment, score)
                    print()
                    """
            if max_score > 0:
                # add newly labeled sentence and the assignment score to the list of sentences in l_max
                u_tagged = assign(max_alignment, max_l, u)
                max_l.add_sentence((u_tagged, max_score))

    results = []  # best newly labeled sentences
    for l in labeled:
        for (u_tagged, score) in highest_score_pairs(k, l.newlyLabeled):
            data = get_jason_data(u_tagged)
            results.append(json.dumps(data, indent=4, sort_keys=True, default=lambda x: x.__dict__))

    runtime = time() - start_time
    print("runtime =  " + str(round(runtime, 2)) + " seconds")
    return results


if __name__ == '__main__':
    init()
    app.run(debug=True)

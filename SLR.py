from flask import Flask
from flask import render_template, request, redirect, url_for
from methods import *
from debug import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded/'


@app.route('/')
def load_homepage():
    return redirect(url_for('home'))


@app.route('/home')
def home():
    return render_template("home.html")


@app.route('/home', methods=['POST'])
def upload():
    unlabeled = request.files['unlabeled_file']
    labeled = request.files['labeled_file']
    if request.form['k'] is not '':
        k = int(request.form['k'])
    else:
        k = 2
    if request.form['threshold'] is not '':
        threshold = float(request.form['threshold'])
    else:
        threshold = 0.6

    # save to upload folder
    if unlabeled.filename == '' or labeled.filename == '':
        return 'Error: no input'
    else:
        unlabeled.save(os.path.join(app.config['UPLOAD_FOLDER'], 'unlabeled.txt'))
        labeled.save(os.path.join(app.config['UPLOAD_FOLDER'], 'labeled.zip'))

    results = '\n\n'.join(run(k, threshold))

    """
    # remove files from upload folder
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'unlabeled.txt'))
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'labeled.zip'))
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'labeled/')
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
    """
    return results


def run(k, threshold):
    """
    The main function.
    Returns a list of newly labeled sentences, represented as JSON files
    """

    labeled = parse_labeled_sentences()
    unlabeled = parse_unlabeled_sentences()

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

                    # debug
                    print_domain(l, alignment_domain)
                    print_range(u, alignment_range)
                    print_alignment(alignment, score)
                    print()

            if max_score > 0:
                # add newly labeled sentence and the assignment score to the list of sentences in l_max
                u_tagged = assign(max_alignment, max_l, u)
                max_l.add_sentence((u_tagged, max_score))

    results = []  # best newly labeled sentences
    for l in labeled:
        for (u_tagged, score) in highest_score_pairs(k, l.newlyLabeled):
            data = get_jason_data(u_tagged)
            results.append(json.dumps(data, indent=4, sort_keys=True, default=lambda x: x.__dict__))

    return results


if __name__ == '__main__':
    init()
    app.run(debug=True)


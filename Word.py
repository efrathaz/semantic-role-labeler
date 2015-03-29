__author__ = 'efrathaz'


class Word:

    def __init__(self, id_num, form, lemma, pos, head, relation):
        self.wordId = int(id_num)
        self.form = form.lower()
        self.lemma = lemma.lower()
        self.pos = pos.lower()
        self.head = int(head)
        self.relation = relation.lower()

        # role information
        self.roleName = None
        self.roleId = None
        self.isFee = False

    def set_role(self, fe_name, fe_id):
        self.roleName = fe_name.lower()
        self.roleId = fe_id

    def print_word(self):
        string = "{" + str(self.wordId) + ", " + self.form + ", " + self.pos + ", " + str(self.head) + "}"
        if self.roleName is not None:
            string = string + " - " + self.roleName
        if self.isFee:
            string += " - FEE"
        print(string)

    def to_string(self):
        string = "{" + str(self.wordId) + ", " + self.form + ", " + self.pos + "}"
        if self.roleName is not None:
            string = string + " - " + self.roleName
        if self.isFee:
            string += " - FEE"
        return string
import math
import operator

#from rougescore import rouge_l
import nltk


# note that all metrics are implemented for a single question and we will have to average over all questions to get the final output of performance
# hence I add a provision to __add__ to a Performance object to keep aggregating the results and a __divide__ method to scale it down finally by the
# number of objects


class Performance():
	def __init__(self, args):
		self.args = args
		self.sum_bleu1 = 0.0
		self.sum_bleu4 = 0.0
		self.sum_rouge = 0.0
		self.sum_meteor = 0.0
		self.examples = 0
		self.mrr = 0
		#self.metoer_scorer = MeteorScorer(self.args.meteor_path)



	def computeMetrics(self, prediction, candidates):
		self.prediction = self.clean_output(" ".join(prediction))
		self.candidates = [self.clean_output(" ".join(c)) for c in candidates]
		self.joint_prediction = " ".join(self.prediction)
		self.joint_candidates = [" ".join(c) for c in self.candidates]
		self.all_candidates = [[c] for c in self.joint_candidates]

		self.compute_bleu()
		# self.rouge = self.compute_rouge()
		#self.meteor = self.compute_meteor()

		self.meteor = 0.0
		self.sum_bleu1 += self.bleu1
		self.sum_bleu4 += self.bleu4

		#self.sum_rouge += self.rouge
		#self.sum_meteor += self.meteor







	def compute_bleu(self):
		if len(self.joint_prediction.split()) != 0:
			self.bleu1 = self.BLEU_1([self.joint_prediction], self.all_candidates)
			self.bleu4 = self.BLEU_4([self.joint_prediction], self.all_candidates)


		else:
			self.bleu1 = 0.0
			self.bleu4 = 0.0
			self.bleu = 0.0

	def compute_rouge(self):
		 return rouge_l(self.joint_prediction, self.joint_candidates, 0.5)
		 # rouge = Pythonrouge(summary=[[self.joint_prediction]], reference=[[self.joint_candidates]], summary_file_exist=False,
		 # 					n_gram=0, ROUGE_SU4=False, ROUGE_L=True, f_measure_only=True,
		 # 					recall_only=False, stemming=True, stopwords=True,
		 # 					word_level=True, length_limit=False,
		 # 					use_cf=False, cf=95, scoring_formula='average',
		 # 					resampling=False)
		 # score = rouge.calc_score()
		 # return score['ROUGE-L']

	def compute_meteor(self):
		self.metoer_scorer.set_reference(self.candidates[0])
		score = self.metoer_scorer._reference.score(self.prediction)
		# self.metoer_scorer.kill_process()
		return score

	def compute_mrr(self):
		## assume that only answer 1 is part of the candidates, not answer 2
		try:
			rank = self.joint_candidates.index(self.joint_prediction)
			return 1./(rank + 1)
		except ValueError:
			return 0.

	def clean_output(self, token_string):
		index_EOS = token_string.find("EOS_TOKEN")
		token_string = token_string[:index_EOS].lower().split()

		tokens = [p for p in token_string]
		# remove one full stop at the end
		if len(tokens) > 0 and tokens[-1] == ".":
			tokens = tokens[:-1]
		if len(tokens) == 0:
			return [""]
		else: return tokens

	def count_ngram(self,candidate, references, n):
		clipped_count = 0
		count = 0
		r = 0
		c = 0
		for si in range(len(candidate)):
			# Calculate precision for each sentence
			ref_counts = []
			ref_lengths = []
			# Build dictionary of ngram counts
			for reference in references:
				ref_sentence = reference[si]
				ngram_d = {}
				words = ref_sentence.strip().split()
				ref_lengths.append(len(words))
				limits = len(words) - n + 1
				# loop through the sentance consider the ngram length
				for i in range(limits):
					ngram = ' '.join(words[i:i + n]).lower()
					if ngram in ngram_d.keys():
						ngram_d[ngram] += 1
					else:
						ngram_d[ngram] = 1
				ref_counts.append(ngram_d)
			# candidate
			cand_sentence = candidate[si]
			cand_dict = {}
			words = cand_sentence.strip().split()
			limits = len(words) - n + 1
			for i in range(0, limits):
				ngram = ' '.join(words[i:i + n]).lower()
				if ngram in cand_dict:
					cand_dict[ngram] += 1
				else:
					cand_dict[ngram] = 1
			clipped_count += self.clip_count(cand_dict, ref_counts)
			count += limits
			r += self.best_length_match(ref_lengths, len(words))
			c += len(words)
		if clipped_count == 0:
			pr = 0
		else:
			pr = float(clipped_count) / count
		bp = self.brevity_penalty(c, r)
		return pr, bp

	def clip_count(self,cand_d, ref_ds):
		"""Count the clip count for each ngram considering all references"""
		count = 0
		for m in cand_d.keys():
			m_w = cand_d[m]
			m_max = 0
			for ref in ref_ds:
				if m in ref:
					m_max = max(m_max, ref[m])
			m_w = min(m_w, m_max)
			count += m_w
		return count

	def best_length_match(self,ref_l, cand_l):
		"""Find the closest length of reference to that of candidate"""
		least_diff = abs(cand_l - ref_l[0])
		best = ref_l[0]
		for ref in ref_l:
			if abs(cand_l - ref) < least_diff:
				least_diff = abs(cand_l - ref)
				best = ref
		return best

	def brevity_penalty(self,c, r):
		if c > r:
			bp = 1
		else:
			bp = math.exp(1 - (float(r) / c))
		return bp

	def geometric_mean(self,precisions):
		return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

	def BLEU_4(self,candidate, references):
		precisions = []
		n = min(4,len(candidate[0].split()))
		pr, bp = self.count_ngram(candidate, references, n)
		precisions.append(pr)
		bleu = self.geometric_mean(precisions) * bp
		return bleu

	def BLEU_1(self,candidate, references):
		precisions = []
		pr, bp = self.count_ngram(candidate, references, 1)
		precisions.append(pr)
		bleu = self.geometric_mean(precisions) * bp
		return bleu



def main():
	pass


if __name__ == "__main__":
	main()

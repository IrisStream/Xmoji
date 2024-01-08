# Data 

## Abstract
**Original data**: [Celebrity Profiling](https://aclanthology.org/P19-1249/) - Celebrities are among the most prolific users of social media, promoting their personas and rallying followers

**Abstraction**: For its construction the Twitter feeds of 71,706 verified accounts have been carefully linked with their respective Wikidata items, crawling both. After cleansing, the resulting profiles contain an average of 29,968 words per profile and up to 239 pieces of personal information. A cross-evaluation that checked the correct association of Twitter account and Wikidata item revealed an error rate of only 0.6%, rendering the profiles highly reliable.

## Analysis

1. Rules to generate name candidates for Wikidata matching from Twitter reference and display names.

	| **ID** | **Name candidate generation rule**               |
	|--------|--------------------------------------------------|
	| I      | Only alphanumeric characters of the display name |
	| II     | Reference name split at capitalization           |
	| III    | Reference name split at display name             |
	| IV     | First and last part from I, split at spaces      |
	| V      | All but the last part from I                     |
	| VI     | All but the last two parts from I                |

2. Evaluation of matching success as per generation rule

	| **ID** | **Success** |
	|--------|-------------|
	| I      | 91.8%       |
	| II     | 2.8%        |
	| III    | > 0.1%      |
	| IV     | 1.8%        |
	| V      | 2.9%        |
	| VI     | 0.3%        |
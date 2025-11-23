CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node 2 \
  longeval/eval_tokenbutler.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --architecture llama \
  --datalen 16384 \
  --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,\
                ruler/niah_multikey_1,ruler/niah_multikey_2,\
                ruler/niah_multiquery,ruler/niah_multivalue,\
                ruler/multiturn_1,ruler/multiturn_2,\
                ruler/fwe,ruler/qa_1,ruler/qa_2,ruler/vt" \
  --dDash 32 \
  --intdim 1024 \
  --result_dir results_tokbutler \
  --eval_llm_mode ExpPred \
  --token_sparse_method fixed_65pc \
  --min_sparse_index 256 \
  --sliding_window 512 \
  --predictor_ckpt /mnt/home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt 



# 
# 


CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node 2 \
  longeval/eval_tokenbutler.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --architecture llama \
  --datalen 16384 \
  --dataset_name "long_bench/qasper" \
  --dDash 32 \
  --intdim 1024 \
  --result_dir results_tokbutler/shortm \
  --eval_llm_mode ExpPred \
  --token_sparse_method fixed_4096tok \
  --min_sparse_index 128 \
  --sliding_window 4096 \
  --tokenbutler_project \
  --predictor_ckpt /home/ya255/projects/TokenButler/expt_model/TokenButler_14Nov_42_finetune_None_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8BiLong_project_4xQMP.csv_L3_8BiLong_project_4xQMP_True_False_2000_False_custom_mix_long_16384_1_1_1_dc2fda83/ExpPred_fixed_40pc_False_False_0_256_False_False_True_False_False_None_False_False_4_8_2_32_1024_False_False_True_False_False_True_tokenbutler_project_32_0.3875000000000002_20251121-015221.pt

  # --dataset_name "ruler/niah_multikey_2" \

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node 2 \
  longeval/eval_tokenbutler.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --architecture llama \
  --datalen 16384 \
  --dataset_name "ruler/niah_multikey_2" \
  --dDash 32 \
  --intdim 1024 \
  --result_dir results_tokbutler/shortm \
  --eval_llm_mode ExpPred \
  --token_sparse_method fixed_65pc \
  --min_sparse_index 256 \
  --sliding_window 512 \
  --predictor_ckpt /mnt/home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt 

# | model                                 | dataset               |   baseline |   samples |
# |:--------------------------------------|:----------------------|-----------:|----------:|
# | meta-llama/Meta-Llama-3.1-8B-Instruct | ruler/niah_multikey_2 |    0.34375 |        96 |
# | mean                                  | mean                  |    0.34375 |        96 |

# | model                                 | dataset               |   baseline |   samples |
# |:--------------------------------------|:----------------------|-----------:|----------:|
# | meta-llama/Meta-Llama-3.1-8B-Instruct | ruler/niah_multikey_2 |   0.583333 |        96 |
# | mean                                  | mean                  |   0.583333 |        96 |

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node 2 \
  longeval/eval_tokenbutler.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --architecture llama \
  --datalen 16384 \
  --dataset_name "ruler/niah_multikey_2" \
  --dDash 32 \
  --intdim 1024 \
  --result_dir results_tokbutler/longm \
  --eval_llm_mode ExpPred \
  --token_sparse_method fixed_65pc \
  --min_sparse_index 256 \
  --sliding_window 16000 \
  --predictor_ckpt /mnt/home/ya255/projects/TokenButler/expt_model/TokenButler_14Nov_42_finetune_None_None__mnt_home_ya255_projects_TokenButler_checkpoints_TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_F_cb1cc44e/4_1000_ExpPred_fixed_40pc_False_False_0_False_False_True_False_False_None_False_False_4_8_2_32_1024_False_False_True_32_0.3875000000000002_20251119-143523.pt

# | model                                 | dataset           |   baseline |   samples |
  # --predictor_ckpt /mnt/home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt 
# |:--------------------------------------|:------------------|-----------:|----------:|
# | meta-llama/Meta-Llama-3.1-8B-Instruct | long_bench/qasper |    0.19062 |       200 |
# | mean                                  | mean              |    0.19062 |       200 |

  # --predictor_ckpt /mnt/home/ya255/projects/TokenButler/expt_model/TokenButler_14Nov_42_finetune_None_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8BiLong_InitV2_PairCE.csv_L3_8BiLong_InitV2_PairCE_True_False_2000_False_custom_mix_long_16384_1_1_1_34677806/4_1000_ExpPred_fixed_40pc_False_False_0_False_False_False_False_True_None_False_False_4_8_2_32_1024_False_False_True_32_0.3875000000000002_20251119-193841.pt
# | model                                 | dataset           |   baseline |   samples |
# |:--------------------------------------|:------------------|-----------:|----------:|
# | meta-llama/Meta-Llama-3.1-8B-Instruct | long_bench/qasper |    0.22314 |       200 |
# | mean                                  | mean              |    0.22314 |       200 |

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node 2 \
  longeval/eval_tokenbutler.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --architecture llama \
  --datalen 16384 \
  --dataset_name "long_bench/qasper" \
  --dDash 32 \
  --intdim 1024 \
  --result_dir results_tokbutler/longm \
  --eval_llm_mode ExpPred \
  --token_sparse_method fixed_65pc \
  --min_sparse_index 128 \
  --sliding_window 512 \
  --predictor_ckpt /mnt/home/ya255/projects/TokenButler/expt_model/TokenButler_14Nov_42_finetune_None_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8BiLong_InitV2_PairCE.csv_L3_8BiLong_InitV2_PairCE_True_False_2000_False_custom_mix_long_16384_1_1_1_34677806/4_1000_ExpPred_fixed_40pc_False_False_0_False_False_False_False_True_None_False_False_4_8_2_32_1024_False_False_True_32_0.3875000000000002_20251119-193841.pt


  # --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,\
  #               ruler/niah_multikey_1,ruler/niah_multikey_2,\
  #               ruler/niah_multiquery,ruler/niah_multivalue,\
  #               ruler/multiturn_1,ruler/multiturn_2,\
  #               ruler/fwe,ruler/qa_1,ruler/qa_2,ruler/vt" \

python demo_gen.py \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --architecture llama \
  --eval_llm_mode ExpPred \
  --token_sparse_method fixed_40pc \
  --model_load_path /mnt/home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt \
  --prompt "In a gentle valley surrounded by low hills, there lay the Kingdom of Myradon. For centuries, its people lived in relative harmony, tilling the land and tending to herds of livestock. They built modest homes, some of timber and some of stone, and each generation passed its skills to the next. The monarchy, established in ancient times, was led by King Adrien, a thoughtful ruler known for his calm demeanor. Every season, local farmers brought fresh produce to the bustling markets of the capital city, hoping to please both their neighbors and the royal court. It was a simple life. in Myradon was marked by a deep respect for the changing seasons. Each year, the spring rains brought new growth. Summer offered steady sunshine to ripen the fields, while autumn ushered in a splendid harvest. Winter was often harsh but gave everyone the chance to gather indoors and share fireside tales. In the main city, known as Highvale, citizens strolled through cobblestone streets that twisted around a gently sloping hill. At its peak, the grand castle presided over all, its towers visible from miles away. Trade caravans made their way along roads that fanned out in every direction.3 King Adrien had inherited the throne from his father, the late King Theodric, whose proud portrait still hung in the royal hall. Despite the comfortable traditions his father had upheld, Adrien sensed that the world was changing. Beyond the mountains to the east, distant realms grew restless, and rumors of shifting alliances began to reach Myradon. While Myradon had weathered minor skirmishes in the distant past, it had not faced significant threats for decades. Many courtiers believed this peace would last forever, yet Adrien felt a subtle tension in the air. He resolved to remain watchful.4 The people admired Adriens open nature. He frequently wandered through Highvale in plain clothing to converse with shopkeepers and artisans. On these walks, he listened to citizens suggestions, even if they seemed trivial. He believed that trust between ruler and subjects formed the strongest bond a kingdom could have. Though the crown still held formal power, many decisions were made in consultation with local leaders. These village elders and city councilors became close advisors to the king, keeping him informed of local disputes, crop conditions, and economic prospects. In return, Adriens fair judgments won him nearly universal support.5 Highvale was not just a seat of power; it was also a cultural center. Musicians, storytellers, and traveling theater troupes entertained the citizens in its main square. Visitors from allied lands brought exotic instruments and shared vibrant melodies that blended with local tunes. Merchants sold handcrafted jewelry, vibrant tapestries, and exotic spices from the southern deserts. Some Myradonians traveled widely, bringing new insights back home. As a result, the kingdom enjoyed a steady stream of cultural exchange. Still, the tranquil routines of daily life remained largely unchanged.6 In that era, however, Myradons peace was not to be taken for granted. Reports began filtering in from scouts who had traveled beyond the western forests. Evidently, a band of raiders was seizing trade caravans in the borderlands. At first, these were small-scale attacks, involving just a few armed rogues. But over time, the raids grew bolder. Merchants who had once journeyed confidently along the trade routes hesitated to leave the safety of the towns. Whispers grew among the populace: could these raiders be a sign of something bigger? Or were they just a nuisance that could be managed?7 King Adrien consulted his chief advisor, Lady Iseryn, about the matter. She was a skilled diplomat known throughout the kingdom for her perceptive mind. Lady Iseryn had once studied in the foreign courts of the Eshenian Confederacy, learning the art of negotiation and the intricacies of forming alliances. She worried that if Myradon did not address the raids soon, this lawlessness might attract opportunists from beyond the region. Adrien convened the Royal Council, and together they decided to dispatch a contingent of the kings guard to reinforce border defenses. They also sent emissaries to neighboring rulers, seeking cooperation.8 Meanwhile, in the quiet town of Breezewood, a short distance from the site of recent raids, farmers and artisans went about their work with an undercurrent of anxiety. Breezewood was home to about a hundred families, many of whom had never encountered serious violence in their lives. The towns leader, Elder Bram, urged people to remain calm and go about their routines. At the same time, he reached out to Highvale, asking for some protective presence of the royal guard. He also asked the townsfolk to keep watch on unfamiliar faces passing through.9 Young Evander, a blacksmiths apprentice in Breezewood, had dreams of knighthood. He practiced daily with makeshift wooden swords. His mentor, the old blacksmith Cedric, observed the boys enthusiasm with a mix of pride and caution. Cedric remembered distant tales of war, told by his own grandfather. He had no desire to see conflict return, yet he recognized that change might be inevitable. He decided to sharpen Evanders mind as well as his swordsmanship, teaching him about discipline and the need for sound judgment. A knight is not just a warrior, Cedric would often say. Hes also a protector. The summer solstice arrived, bringing with it a grand festival throughout Myradon. In Highvale, bright pennants fluttered from windows, and the squares were filled with dancing. Music echoed across the city streets, and traveling performers amused everyone with acrobatics and comedic sketches. King Adrien took this opportunity to address the crowd from a balcony overlooking the castle courtyard. He assured them that the raids in the west would soon be put to an end. He spoke of unity, cooperation, and preserving the kingdoms cherished way of life. His words instilled hope in the hearts of many who listened.In the days following the festival, rumors emerged that the raiders were actually part of a larger force, once loyal to a fallen noble house. Some said that the band was led by a man named Braxis, who supposedly had a grudge against Myradons monarchy. Others suggested he was merely a ruthless opportunist trying to carve out a personal domain at the edge of civilization. Whatever the truth, these stories spread quickly, fanned by the fear of caravans disappearing without a trace. Merchants delayed their journeys, uncertain if they should risk the roads. At the royal castle, Lady Iseryn organized a small conference with military and diplomatic representatives from neighboring territories. Delegates arrived from the Riverlands to the east, from the mountainous domain of Torlith in the north, and from the plains of Arneth in the south. Each shared intelligence on the bandit movements and potential threats. While these lands had no formal obligation to intervene, they valued trade with Myradon and did not wish to see the region descend into chaos. As a result, the meeting concluded with a promise of limited cooperation against any threat that might spread. To reinforce Myradons stability, the king also put forth new economic measures. He arranged for small subsidies for farmers who lost goods to bandit raids, hoping to keep them from financial ruin. At the same time, he encouraged merchant guilds to hire skilled escorts for their caravans. Many young men and women saw this as a chance to earn a living by defending trade routes. In particular, eager volunteers from rural towns signed up for this work, seeking not just pay but also the chance to win renown for themselves and their families. Among those volunteers were Evander and his mentor Cedric. Although Cedric was too old for combat duty, his skill at forging armor and weapons proved invaluable. He joined a guild caravan as the official smith, able to repair any damage the guards gear might sustain. Evander, though just sixteen, secured a position as a junior guard, trained to watch for signs of ambush and to assist more experienced fighters. It was a bold decision, but both saw it as a way to do their part in safeguarding Myradon. Breezewood wished them luck, offering small tokens for good fortune. Their first journey took them along the Old Stone Road, a route that snaked westward through dense woodlands. The caravan moved slowly but steadily, flanked by a dozen armed riders and a few wagons of supplies. Evander kept a close eye on the treeline, remembering Cedrics warnings about ambushes. Nights were spent in makeshift camps, with watch rotations ensuring everyone got some rest. Despite occasional rustling in the darkness, the caravan passed through without incident. They eventually arrived at the small fortress of Stonecross, a border outpost where travelers often stopped for fresh provisions and information on local threats. At Stonecross, the group found that a handful of recent travelers had been attacked by bandits a few days prior. Most had lost valuables, though they escaped with their lives. The bandits melted into the forest before any patrols could respond, leaving little trace behind. The fortress commander, a pragmatic soldier named Captain Roswyn, briefed the newcomers on the situation. She mentioned that the attackers seemed more organized than typical rogues, with a lookout system and a hierarchy of command. Still, they had not yet mustered enough force to overwhelm heavily guarded convoys. Evander listened carefully to Roswyns advice on defensive strategies, especially the use of scouts and archers. If a fight broke out, the attackers often relied on shock and confusion, so watchful eyes were the best deterrent. Soon after, Cedric and Evanders caravan resumed its westward trip. On the second night beyond Stonecross, they encountered their first real test. A band of raiders emerged from the shadows at dusk, shouting threats and demanding the caravans goods. Evanders heartbeat thundered, but he steadied himself behind a sturdy shield, ready to protect the wagons. In the light of the campfire, a fierce clash erupted. Arrows whistled through the air, and steel clanged against steel. Cedric, though not a primary fighter, stood by with a hammer in hand, prepared to defend himself if necessary. The caravan guards had formed a defensive perimeter, showing the training and coordination that Captain Roswyn had emphasized. Evander managed to block a blow aimed at one of the merchant drivers. His counterstrike was not lethal, but it forced his attacker to retreat. After a few tense minutes, the raiders pulled back, evidently realizing the caravan was too well defended. The incident ended as quickly as it started, but it left a palpable sense of urgency. While no one was severely injured on the caravan side, the bandits sudden appearance showed how vulnerable even an organized group could be at night. Evanders nerves were frayed, yet he also felt a jolt of confidence—he had faced real danger and done his part. Cedric praised his composure, reminding him that fighting was always a last resort but that one must be prepared if peaceful options fail. The rest of the night passed without further attacks. News of the skirmish soon spread across the frontier. At the royal castle, King Adrien received a detailed report through a courier. Though relieved that the caravan survived, he worried about what would come next. He convened a strategic meeting with his generals, including an older knight named Sir Gareth, renowned for his defense of Myradon in years past. Sir Gareth believed that these raids were likely the work of a central figure trying to unite scattered outlaws. Adrien agreed, suspecting that a shadowy leader like Braxis might be behind it. They realized something needed to be done soonWhile preparations for a more robust response continued in Highvale, another challenge quietly brewed in the south. Drought conditions had left several villages near the Arneth border struggling with low water supply. The Myradonian farmers there appealed for help. The kingdom dispatched engineers to dig new wells and devise irrigation improvements, hoping to stave off famine. As resources and attention turned in multiple directions, Adrien found himself juggling urgent matters on several fronts. He understood that a kingdoms strength also depended on addressing everyday problems, not just potential military threats. Back in Breezewood, people remained alert but carried on their daily lives as best they could. Elder Bram held regular meetings in the town square to share updates and reassure his neighbors that the royal guard was actively patrolling nearby roads. Many farmers worked at planting or harvesting, aware that food supplies were critical for morale. Children still played in the meadows, though parents kept a closer eye on them. Anxieties lingered, yet there was a collective determination to not let fear paralyze their community. Cedric and Evanders caravan arrived in the bustling trade post of Hollowford, located near an ancient river. Although somewhat removed from Myradons core, Hollowford was an essential link in the flow of goods to other territories. Armed escorts were commonplace there, creating a sense of guarded vigilance. Merchants swapped stories of raids, comparing notes on the best routes to avoid trouble. At a local tavern, Cedric picked up rumors suggesting Braxis was amassing a following in hidden camps scattered around the western woods. Evander found this both alarming and oddly intriguing, as if a great challenge lay ahead. The pair soon received a letter from Lady Iseryn herself, requesting them to remain in Hollowford for a short while. She believed it was wise to gather reliable witnesses to the bandit threat, and Cedrics firsthand account carried weight. Feeling honored, they awaited further instruction. In the interim, Evander continued to train, practicing sword drills with local guards. Cedric offered blacksmithing services to keep equipment in top shape. This arrangement helped pay for their stay, and it also boosted the readiness of the settlements defenses. Each day that passed, more merchants arrived with new tales of trouble on the roads. Eventually, an official envoy from the palace arrived, led by Sir Gareth himself. He invited Cedric and Evander to join a scouting party tasked with locating one of Braxiss rumored camps. The group included a few skilled rangers, knowledgeable about forest tracks and terrain. Cedric hesitated initially—he was no soldier, and Evander was still quite young. Yet, they both realized how critical reliable information would be for the kingdom. They agreed to join, trusting that Sir Gareths leadership would keep them as safe as possible under the circumstances. Setting out at dawn, the scouting party traveled light, carrying only essential supplies. The rangers guided them along hidden paths that circumvented the more obvious routes. They aimed to observe the bandits movements without directly engaging them unless necessary. For days, they crept through dense undergrowth, careful to leave no trace of their passage. Occasionally, they found abandoned campfires, scattered footprints, and other signs that a group had been there. The tension grew each time they thought they heard distant voices or rustling. Evander, though still nervous, felt more confident with" \
  --dDash 32 \
  --sliding_window 16 \
  --intdim 1024 \
  --max_new_tokens 256

python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 800 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "TEST_GENERATION.csv" \
    --wname TEST_GENERATION \
    --no_wandb \
    --pred_lr 1e-3 \
    --dDash 32 \
    --sliding_window 16 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path /mnt/home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_L3_8B_R1.csv_L3_8B_R1_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt

    


python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 800 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "TEST_GENERATION.csv" \
    --wname TEST_GENERATION \
    --no_wandb \
    --pred_lr 1e-3 \
    --dDash 32 \
    --sliding_window 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path /mnt/home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt
    
    



CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node 2 \
  longeval/eval_tokenbutler.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --architecture llama \
  --datalen 16384 \
  --dataset_name "ruler/niah_multikey_2" \
  --dDash 32 \
  --intdim 1024 \
  --result_dir results_tokbutler/shortm \
  --eval_llm_mode ExpPred \
  --token_sparse_method fixed_65pc \
  --min_sparse_index 256 \
  --sliding_window 512 \
  --tokenbutler_project \
  --predictor_ckpt /mnt/home/ya255/projects/TokenButler/expt_model/TokenButler_14Nov_42_finetune_None_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8BiLong_project.csv_L3_8BiLong_project_True_False_2000_False_custom_mix_long_16384_1_1_10_0.001_8_16_7efef68c/ExpPred_fixed_40pc_False_False_0_256_False_False_True_False_False_None_False_False_4_8_2_32_1024_False_False_True_False_False_True_tokenbutler_project_32_0.3875000000000002_20251120-195257.pt



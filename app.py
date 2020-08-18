
import xpress as xp
import numpy as np
import pandas as pd
import streamlit as st
import math

class scheduler:
    def __init__(self, plyr_list, n_rounds=3, max_gm_plays=2, max_gm_plays_soft=False, trim_gms_for_feasibility=True):
        self.plyr_list = plyr_list
        self.n_plyrs = len(plyr_list)
        self.trim_gms=trim_gms_for_feasibility
        self.n_rounds = n_rounds
        self.n_tms=2
        self.max_gm_gap=1 # max gap b/w number of games played by any two players
        self.max_gm_plays=max_gm_plays # max number of times a player can play any game
        self.max_gm_plays_soft = max_gm_plays_soft
        self.game_list=[('Bocci',1),('Ping_Pong',2), ('Kornhole',2), ('Can_Jam',2)]
    
    def check_model_feasibility(self):
        print('Checking feasibility')
        self.model.iisfirst(1)
        # self.model.iisall() # for LPs only
        if self.model.attributes.numiis == 0:
            pass
        else:
            st.text(
                "Problem is infeasible. Try to relax 1+ constraint from the IIS.".format(
                    self.model.attributes.numiis
                )
            )
            for i in range(1, self.model.attributes.numiis + 1):
                st.text("IIS {} Constraints:".format(i))
                miisrow = []
                miiscol = []
                constrainttype = []
                colbndtype = []
                duals = []
                rdcs = []
                isolationrows = []
                isolationcols = []
                # get data for the first IIS
                self.model.getiisdata(
                    i,
                    miisrow,
                    miiscol,
                    constrainttype,
                    colbndtype,
                    duals,
                    rdcs,
                    isolationrows,
                    isolationcols,
                )
                for k in miisrow:
                    st.text(k.name)

    def solve(self):
        
        ### Params ###
        self.n_plyrs_per_game = [self.game_list[i][1] for i in range(len(self.game_list))]
        req_plyrs_per_round = sum([ct*2 for ct in self.n_plyrs_per_game])
        
        if self.trim_gms & (req_plyrs_per_round > self.n_plyrs):
            st.text("Removing games to fit the number of passed players")
            while req_plyrs_per_round > self.n_plyrs:
                self.game_list = self.game_list[:-1]
                self.n_plyrs_per_game = [self.game_list[i][1] for i in range(len(self.game_list))]
                req_plyrs_per_round = sum([ct*2 for ct in self.n_plyrs_per_game])
            print(req_plyrs_per_round, self.n_plyrs)
            # max number of times a player can play a given game
            self.max_gm_plays=max(2, math.ceil(req_plyrs_per_round*self.n_rounds/self.n_plyrs)) 
        st.text(f'Final game list: {self.game_list}')
        if not self.max_gm_plays_soft: st.text(f'Max times a player is allowed to play same game: {self.max_gm_plays}')
        self.n_games = len(self.game_list)

        ## Instantiate problem ##
        self.model = xp.problem()

        #################
        ### Variables ###
        #################
        # Player-game-round-team
        plyrs={}
        for p in range(self.n_plyrs):
            plyrs[p]={}
            for g in range(self.n_games):
                plyrs[p][g]={}
                for r in range(self.n_rounds):
                    plyr_var_list = [xp.var(name=f"p{p}_g{g}_r{r}_t{t}", vartype=xp.binary) for t in range(self.n_tms)]
                    plyrs[p][g][r]=plyr_var_list
                    self.model.addVariable(plyr_var_list)

        # Team assignments
        tms={}
        for t in range(self.n_tms):
            tm_var_list = [xp.var(name=f"p{p}_tm{t}", vartype=xp.binary) for p in range(self.n_plyrs)]
            tms[t]=tm_var_list
            self.model.addVariable(tm_var_list)
        
        # Variable set greater_than/equal_to game_played count for all players for all games - this quantity is minimized
        max_gm_plays_global = xp.var(name='max_gm_plays_global', lb=0)
        self.model.addVariable(max_gm_plays_global)

        ###################
        ### Constraints ###
        ###################
        # Correct number of plyrs per game per team
        ## Why need both less/greater than and not equal to?
        for g in range(self.n_games):
            for r in range(self.n_rounds):
                for t in range(self.n_tms):
                    suff_plyrs_tm_1 = xp.Sum(plyrs[p][g][r][t] for p in range(self.n_plyrs)) >= self.n_plyrs_per_game[g]
                    suff_plyrs_tm_2 = xp.Sum(plyrs[p][g][r][t] for p in range(self.n_plyrs)) <= self.n_plyrs_per_game[g]
                    self.model.addConstraint( xp.constraint(suff_plyrs_tm_1, name = f'gteq_plyrs_tm{t}_rnd{r}_gm{g}') )
                    self.model.addConstraint( xp.constraint(suff_plyrs_tm_2, name = f'lteq_plyrs_tm{t}_rnd{r}_gm{g}') )

        # One game per time per player
        for p in range(self.n_plyrs):
            for r in range(self.n_rounds):
                for t in range(self.n_tms):
                    one_game_per_round_per_plyr = xp.Sum(plyrs[p][g][r][t] for g in range(self.n_games)) <= 1
                    self.model.addConstraint(xp.constraint(one_game_per_round_per_plyr, 
                                                  name=f"one_gm_in_rnd{r}_for_plyr{p}_tm{t}"))

        # Team assignment constraints
        for p in range(self.n_plyrs):
            # One team per player
            tm_lb = xp.Sum(tms[t][p] for t in range(self.n_tms)) <= 1
            tm_ub = xp.Sum(tms[t][p] for t in range(self.n_tms)) >= 1
            self.model.addConstraint( xp.constraint(tm_lb, name=f'plyr{p}_lteq_1_tm') )
            self.model.addConstraint( xp.constraint(tm_ub, name=f'plyr{p}_gteq_1_tm') )
            for t in range(self.n_tms):
                # Forcing tm variables to be flipped when player selected
                tm_enforce = xp.Sum(plyrs[p][g][r][t] for g in range(self.n_games) for r in range(self.n_rounds)) <= tms[t][p]*self.n_rounds
                self.model.addConstraint( xp.constraint(tm_enforce, name=f'plyr{p}_tm{t}_enforce'))

        # Teams evenly split
        tms_even_1 = xp.Sum(tms[0][p] for p in range(self.n_plyrs)) - xp.Sum(tms[1][p] for p in range(self.n_plyrs)) <= 1
        tms_even_2 = xp.Sum(tms[1][p] for p in range(self.n_plyrs)) - xp.Sum(tms[0][p] for p in range(self.n_plyrs)) <= 1
        self.model.addConstraint(tms_even_1)
        self.model.addConstraint(tms_even_2)

        # Each player plays each game at most 'self.max_gm_plays'
        for p in range(self.n_plyrs):
            for g in range(self.n_games):
                for t in range(self.n_tms):
                    max_rds_per_game_per_plyr = xp.Sum(plyrs[p][g][r][t] for r in range(self.n_rounds)) 
                    if not self.max_gm_plays_soft:
                        self.model.addConstraint( xp.constraint(max_rds_per_game_per_plyr <= self.max_gm_plays , 
                                                       name=f"plyr{p}_plays_gm{g}_max_{self.max_gm_plays}_times_tm{t}"))
                    self.model.addConstraint( xp.constraint(max_gm_plays_global >= max_rds_per_game_per_plyr,
                                                   name=f'max_gm_plays_global_gteq_p{p}_g{g}_t{t}'))

        # Each player plays at most once more than every other player
        for p1 in range(self.n_plyrs):
            n_plays = xp.Sum(plyrs[p1][g][r][t] for g in range(self.n_games) for r in range(self.n_rounds) for t in range(self.n_tms))
            for p2 in range(p1+1, self.n_plyrs):
                n_plays_ = xp.Sum(plyrs[p2][g][r][t] for g in range(self.n_games) for r in range(self.n_rounds) for t in range(self.n_tms))
                self.model.addConstraint( (n_plays - n_plays_) <= self.max_gm_gap)
                self.model.addConstraint( (n_plays_ - n_plays) <= self.max_gm_gap)

        # Objective
        self.model.setObjective(max_gm_plays_global, sense=xp.minimize)
        self.model.solve()
        self.check_model_feasibility()
        return plyrs, tms
        
        
    def postprocess_solution(self, plyrs, tms):

        # Compiling variable results
        vars=[]
        for p in plyrs.keys():
            for g in plyrs[p].keys():
                for r in plyrs[p][g]:
                    for t in plyrs[p][g][r]:
                        vars.append(t)
        for t in tms.keys(): vars.extend(tms[t])
        assert len(self.model.getSolution()) == len(vars)+1 # +1 for max_plays_global var
        self.max_gm_plays = self.model.getSolution()[-1]
        st.text(f'Max times a player plays the same game: {int(self.max_gm_plays)}')

        # Creating df to analyze results
        df=pd.DataFrame(list(zip(vars, self.model.getSolution())))
        df.columns=['var','val']
        df['var_str']=df['var'].astype(str)

        # Ensuring each team has proper player count
        tm0_plyr_ct = df[df.var_str.apply(lambda x : 'tm0' in x)]['val'].sum()
        assert np.abs(self.n_plyrs/2 - tm0_plyr_ct) <= 0.5, print(self.n_plyrs/2, df[df.var_str.apply(lambda x :'tm' in x)]['val'].sum())

        ### Players ###
        plyr_df = df[df.var_str.apply(lambda x : 'tm' not in x)]
        plyr_df['plyr']=plyr_df['var_str'].apply(lambda x : x.split('_')[0][1:])
        plyr_df['game']=plyr_df['var_str'].apply(lambda x : x.split('_')[1][1])
        plyr_df['rnd']=plyr_df['var_str'].apply(lambda x : x.split('_')[2][1])
        plyr_df['tm']=plyr_df['var_str'].apply(lambda x : x.split('_')[3][1])

        print(f"Total players per round:\n {plyr_df.groupby('rnd')['val'].sum().reset_index()}")

        # Ensuring each player plays within self.max_gm_gap games of player with max games played
        plyr_gm_cts = plyr_df.groupby('plyr')['val'].sum().reset_index()
        print(f"Diff b/w max plays and min plays for single player: {max(plyr_gm_cts.val) - min(plyr_gm_cts.val)}")
        assert max(plyr_gm_cts.val) - min(plyr_gm_cts.val) <= self.max_gm_gap

        # Ensuring each player plays for one team
        tm_cts_per_plyr = plyr_df.groupby(['plyr','tm'])['val'].sum().reset_index()
        self.n_tms_per_plyr = tm_cts_per_plyr.groupby('plyr').apply(lambda df : (df.val > 0).sum() ).reset_index()
        assert (self.n_tms_per_plyr[0].values == 1).sum() == self.n_tms_per_plyr.shape[0]

        # Ensuring each player plays each game <= self.max_gm_plays times
        game_cts_per_plyr = plyr_df.groupby(['plyr','game'])['val'].sum().reset_index()
        assert (game_cts_per_plyr.val.values <= self.max_gm_plays).sum() == game_cts_per_plyr.shape[0]

        # Ensuring each player plays each game <= self.max_gm_plays times
        rnd_cts_per_plyr = plyr_df.groupby(['plyr','rnd'])['val'].sum().reset_index()
        assert (rnd_cts_per_plyr.val.values <= 1).sum() == rnd_cts_per_plyr.shape[0]

        ### Postprocessing DF
        plyr_df['Player'] = plyr_df['plyr'].apply(lambda x : self.plyr_list[int(x)])
        plyr_df['Game'] = plyr_df['game'].apply(lambda x : self.game_list[int(x)][0])
        plyr_df['Round'] = plyr_df['rnd'].astype(int)+1
        plyr_df['Team'] = plyr_df['tm'].apply(lambda x : 'Red' if int(x) == 0 else 'Blue')
        res = plyr_df[plyr_df.val==1].sort_values(['rnd','game','tm'])
        res.reset_index(inplace=True, drop=True)
        return res[['Player','Round','Game','Team']]
              
    
def test_scheduler():
    n_rounds = 3
    plyr_list = ['Mike', 'Jenna', 'Miller', 'Jackie', 'Brock', 'Janelle', 'Laura', 'Dane'] #, Jarrod, Renee, Leif, Glen, Nate']
    s = scheduler(plyr_list, n_rounds=n_rounds, trim_gms_for_feasibility=True, max_gm_plays=2)
    plyrs, tms = s.solve()
    res = s.postprocess_solution(plyrs, tms)
#     df=test_scheduler()
    return res

if __name__ == '__main__':
    st.title('2021 Game Scheduler')
    
    # Getting number of rounds from user
    n_rounds = st.text_area("Enter number of rounds of games\n")
        
    # Getting player list from user
    st.text('Copy the names below as an example or enter your own names')
    st.text('Mikey, Janet, Miyerr, Jacqueline, Brokk, Jalen') 
    plyr_list = st.text_area("Enter player names separated by commas\n")
    plyr_list = plyr_list.split()
    plyr_list = [plyr.replace(',','') for plyr in plyr_list]
    
    # Proceed once players captured
    if len(plyr_list) > 0:
        try: n_rounds = int(n_rounds)
        except: 
            n_rounds=3
            st.text("Using default value of 3 rounds")
        s = scheduler(plyr_list, n_rounds=n_rounds, trim_gms_for_feasibility=True, max_gm_plays=1)
        plyrs, tms = s.solve()
        if s.model.attributes.numiis > 0:
            s = scheduler(plyr_list, n_rounds=n_rounds, trim_gms_for_feasibility=True, max_gm_plays=2)
            plyrs, tms = s.solve()
        res = s.postprocess_solution(plyrs, tms)
        st.table(res)

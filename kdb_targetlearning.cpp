/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 ** Implements Sahami's k-dependence Bayesian classifier
 **
 ** This program is free software: you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation, either version 3 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 ** GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program. If not, see <http://www.gnu.org/licenses/>.
 **
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "kdb_targetlearning.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

kdb_targetlearning::kdb_targetlearning() : pass_(1) //？？
{
}

kdb_targetlearning::kdb_targetlearning(char*const*& argv, char*const* end) : pass_(1)
{
    //printf("init\n");
    name_ = "kdb_targetlearning";

    // defaults
    k_ = 1;

    // get arguments
    while (argv != end)
    {
        if (*argv[0] != '-')
        {
            break;
        } else if (argv[0][1] == 'k')
        {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else
        {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
    // 仅考虑k=2的情况
    assert(k_==2);
    mode = 1;
    // 0 for locals
    // 1 gens and locals
}

kdb_targetlearning::~kdb_targetlearning(void)
{
}

void kdb_targetlearning::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass
{
public:

    miCmpClass(std::vector<float> *m)
    {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b)
    {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};
class local_miCmpClass
{
public:

    local_miCmpClass(std::vector<double> *m)
    {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b)
    {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<double> *mi;
};
//计算两个属性变量与类变量结点C之间的互信息，返回的是互信息最大的那个属性结点

void kdb_targetlearning::reset(InstanceStream &is)
{
    //printf("reset\n");
    //printf("reset\n");
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1
    //K_表示属性结点可以作为父节点的结点的个数
    // initialise distributions
    dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);


    for (CategoricalAttribute a = 0; a < noCatAtts; a++)
    {
        parents_[a].clear(); //？？
        dTree_[a].init(is, a); //DTree   used in the second pass and for classification
    }
    //Insts.resize(0);
    /*初始化各数据结构空间*/
    xxxyDist_.reset(is);
    dist_.reset(is); //xxyDist dist_;

    classDist_.reset(is); //yDist classDist_;

    pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
void kdb_targetlearning::update_local_dTrees(){
    return;
    local_dTrees.resize(this->noClasses_);
    for(int y=0;y<this->noClasses_;y++){
        local_dTrees[y].resize(this->noCatAtts_);
        for (CategoricalAttribute a = 0; a < this->noCatAtts_; a++)
        {
            local_dTrees[y][a].init(* this->instanceStream_, a); //DTree   used in the second pass and for classification
        }
    }
    for(int i=0;i<this->Insts.size();i++){
        for(int y=0;y<this->noClasses_;y++){
            for (CategoricalAttribute a = 0; a < this->noCatAtts_; a++)
            {
                local_dTrees[y][a].update(this->Insts[i], a, local_parents[y][a]);
            }
        }
    }

}
/*通过训练集来填写数据空间*/
void kdb_targetlearning::train(const instance &inst)
{
    if (pass_ == 1)
    {
        dist_.update(inst);//只更新xxyDist
        xxxyDist_.update(inst);
        //Insts.push_back(inst);
    }
    else
    {
        assert(pass_ == 2);

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
            dTree_[a].update(inst, a, parents_[a]);
        }
        classDist_.update(inst);
    }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void kdb_targetlearning::initialisePass()
{
}

void kdb_targetlearning::classify_local(const instance& inst,std::vector<std::vector<CategoricalAttribute> > parrent,std::vector<double> &posteriorDist)
{
    // calculate the class probabilities in parallel
    // P(y)
    for (CatValue y = 0; y < noClasses_; y++)
    {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }


    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        // const CategoricalAttribute parent1 = parents_1[x1][0];//typedef CategoricalAttribute parent unsigned int
        // const CategoricalAttribute parent2 = parents_1[x1][1];

        if (parrent[x1].size()==0) {
            //printf("PARent=0  \n");
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist[y] *=xxxyDist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);  // p(a=v|Y=y) using M-estimate
                //printf("x1=%d     y=%d\n",x1,y);
                //printf("PARent=0  :%f\n",xxxyDist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y));
            }
        }
        else if(parrent[x1].size()==1){
            //printf("PARent=1  \n");
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist[y] *=xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1), parrent[x1][0],inst.getCatVal(parrent[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                //classDist_2[y] *=xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1), parent12,inst.getCatVal(parent12), y);
                //printf("x1=%d       parents_1[x1][0]=%d     y=%d\n",x1,parrent[x1][0],y);
                //printf("PARent=1  :%f\n",xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1), parrent[x1][0],inst.getCatVal(parrent[x1][0]), y));
            }
        }
        else if(parrent[x1].size()==2){
            //printf("PARent=2  \n");
            for (CatValue y = 0; y < noClasses_; y++) {// p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
               //printf("x1=%d       parents_1[x1][0]=%d     parents_1[x1][1]=%d     y=%d\n",x1,parrent[x1][0],parrent[x1][1],y);
               //printf("PARent=2  :%f\n",xxxyDist_.p(x1, inst.getCatVal(x1), parrent[x1][0],inst.getCatVal(parrent[x1][0]),parrent[x1][1],inst.getCatVal(parrent[x1][1]), y));
               posteriorDist[y] *= xxxyDist_.p(x1, inst.getCatVal(x1), parrent[x1][0],inst.getCatVal(parrent[x1][0]),parrent[x1][1],inst.getCatVal(parrent[x1][1]), y);
               //classDist_2[y] *= dist_.p(x1, inst.getCatVal(x1), parent12,inst.getCatVal(parent12),parent22,inst.getCatVal(parent22), y);

            }
        }
    }

    // normalise the results
    normalise(posteriorDist);
}


/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void kdb_targetlearning::finalisePass()
{
    //printf("finalisePass\n");
    if (pass_ == 1 && k_ != 0)
    {
        // calculate the mutual information from the xy distribution
        std::vector<float> mi;
        getMutualInformation(dist_.xyCounts, mi);//计算好了I(Xi;C)

        if (verbosity >= 3)
        { //？
            printf("\nMutual information table\n");
            print(mi);
        }

        // calculate the conditional mutual information from the xxy distribution
        crosstab<float> cmi = crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_, cmi);

        //dist_.clear();

        if (verbosity >= 3)
        {
            printf("\nConditional mutual information table\n");
            cmi.print();
        }

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order; //？？order存放的是所有的属性结点

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
            order.push_back(a);
        }

        // assign the parents
        if (!order.empty())
        {
            miCmpClass cmp(&mi);

            std::sort(order.begin(), order.end(), cmp); //？？

            if (verbosity >= 2)
            {
                printf("\n%s parents:\n", instanceStream_->getCatAttName(order[0]));
            }

            // proper kdb_targetlearning assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++)
            {
                parents_[*it].push_back(order[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++)
                {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (parents_[*it].size() < k_)
                    {
                        // create space for another parent
                        // set it initially to the new parent.
                        // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        parents_[*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < parents_[*it].size(); i++)
                    {
                        if (cmi[*it2][*it] > cmi[parents_[*it][i]][*it])
                        {
                            // move lower value parents down in order
                            for (unsigned int j = parents_[*it].size() - 1; j > i; j--)
                            {
                                parents_[*it][j] = parents_[*it][j - 1];
                            }
                            // insert the new att
                            parents_[*it][i] = *it2;
                            break;
                        }
                    }
                }


            }
        }


    }

    ++pass_;
}

// true if no more passes are required. updated by finalisePass()

bool kdb_targetlearning::trainingIsFinished()
{
    return pass_ > 2;
}

void kdb_targetlearning::classify(const instance& inst, std::vector<double> &posteriorDist)
{
    local_parents.resize(this->noClasses_);
    //local_dTrees.resize(this->noClasses_);
    for(int y=0;y<this->noClasses_;y++){
        local_parents[y].resize(this->noCatAtts_);
        //local_dTrees[y].resize(noCatAtts);
        for (CategoricalAttribute a = 0; a < this->noCatAtts_; a++)
        {
            local_parents[y][a].clear(); //？？
            //local_dTrees[y][a].init(is, a); //DTree   used in the second pass and for classification
        }
    }
    //printf("classify\n");
    for(int y =0;y<this->noClasses_;y++){
        std::vector<double> mi;
        mi.resize(this->noCatAtts_);
        for(int x=0;x<this->noCatAtts_;x++){
            mi[x] = dist_.xyCounts.jointP(x,inst.getCatVal(x),y)
                             *
                    log(
                        dist_.xyCounts.jointP(x,inst.getCatVal(x),y)
                                /
                        (dist_.xyCounts.p(x,inst.getCatVal(x)) * dist_.xyCounts.p(y))
                    );

        }
        std::vector< std::vector<double> >cmi;
        cmi.resize(this->noCatAtts_);
        for(int x1 =0;x1<this->noCatAtts_;x1++){
            cmi[x1].resize(this->noCatAtts_);
            for(int x2=0;x2<this->noCatAtts_;x2++){
                if(x1==x2){
                    cmi[x1][x2]=0;
                    continue;
                }
                cmi[x1][x2]=dist_.jointP(x1,inst.getCatVal(x1),x2,inst.getCatVal(x2),y)
                                    *
                            log(
                                (dist_.jointP(x1,inst.getCatVal(x1),x2,inst.getCatVal(x2),y)/dist_.xyCounts.p(y))
                                                     /
                                (
                                   dist_.xyCounts.p(x1,inst.getCatVal(x1),y) * dist_.xyCounts.p(x2,inst.getCatVal(x2),y)
                                )


                            );
            }
        }
        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order; //？？order存放的是所有的属性结点

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
            order.push_back(a);
        }

        // assign the parents
        if (!order.empty())
        {
            local_miCmpClass cmp(&mi);

            std::sort(order.begin(), order.end(), cmp); //？？

            // proper KDB assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++)
            {
                local_parents[y][*it].push_back(order[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++)
                {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (local_parents[y][*it].size() < k_)
                    {
                            // create space for another parent
                            // set it initially to the new parent.
                            // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        local_parents[y][*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < local_parents[y][*it].size(); i++)
                    {
                        if (cmi[*it2][*it] > cmi[local_parents[y][*it][i]][*it])
                        {
                            // move lower value parents down in order
                            for (unsigned int j = local_parents[y][*it].size() - 1; j > i; j--)
                            {
                                local_parents[y][*it][j] = local_parents[y][*it][j - 1];
                            }
                            // insert the new att
                            local_parents[y][*it][i] = *it2;
                            break;
                        }
                    }
                }
            }
        }

    }



    // calculate the class probabilities in parallel
    // P(y)
    std::vector<double> genDist;
    genDist.resize(this->noClasses_);
    for (CatValue y = 0; y < noClasses_; y++)
    {
        genDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }

    // P(x_i | x_p1, .. x_pk, y)
    for (CategoricalAttribute x = 0; x < noCatAtts_; x++)
    {

        dTree_[x].updateClassDistribution(genDist, x, inst);
    }

    // normalise the results
    normalise(genDist);

    //update_local_dTrees();

    std::vector<double> local_final_Dist(this->noClasses_,0.0);
    for(int y=0;y<this->noClasses_;y++){
        std::vector<double> localDist;
        localDist.resize(this->noClasses_);
        classify_local(inst,this->local_parents[y],localDist);

        for (CatValue yi = 0; yi < noClasses_; yi++){
            local_final_Dist[yi]+=localDist[yi];
        }
    }
    normalise(local_final_Dist);
    if(mode ==0){
        for (CatValue yi = 0; yi < noClasses_; yi++){
            posteriorDist[yi]=local_final_Dist[yi];
        }
    }
    else{
        for (CatValue yi = 0; yi < noClasses_; yi++){
            posteriorDist[yi]=local_final_Dist[yi]+genDist[yi];
        }

    }
    normalise(posteriorDist);

}




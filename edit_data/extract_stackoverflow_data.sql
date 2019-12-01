select Id, Score, ViewCount, Body, LastEditorUserId, OwnerUserId, Title,
Tags, AnswerCount, CommentCount, FavoriteCount
from Posts
where PostTypeId = 1
and CreationDate > '2018-12-01'


select ParentId, OwnerUserId
from Posts
where PostTypeID = 2
and CreationDate > '2018-12-01'


select PostId, UserId
from Comments
where CreationDate > '2018-12-01'


select PostId, UserId
from Votes
where CreationDate > '2018-12-01'
